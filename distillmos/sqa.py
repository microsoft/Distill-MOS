import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .third_party.xls_r_sqa.sqa_model import SingleLayerModel
from .third_party.xls_r_sqa.config import Config, FEAT_SEQ_LEN

N_LAYERS_CNN = 6
CNN_FINAL_CHANNELS = 256
N_HEADS = 8
DIM_TRANSFORMER = 200
N_LAYERS_TRANSFORMER = 7
SEQ_LEN = 122880
MAX_HOP_LEN = 16000

thispath = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS_CHKPT = os.path.join(thispath, "weights", "distill_mos_v7.pt")


def complex_compressed(x, hop_length, win_length):
    n_fft = win_length
    x = F.pad(
        x,
        (int((win_length - hop_length) / 2), int((win_length - hop_length) / 2)),
        mode="reflect",
    )
    x = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=False,
        return_complex=True,
        window=torch.hann_window(win_length).to(x.device),
    )
    x = torch.view_as_real(x)
    mag = torch.sum(x**2, dim=-1, keepdim=True) ** 0.5
    compressed = torch.clip(mag, 1e-12) ** 0.3 * x / torch.clip(mag, 1e-12)
    return compressed


class ComplexSpecCNN(nn.Module):
    def __init__(self, final_channels, num_layers):
        super().__init__()
        self.hop_length = 160  # 10 ms
        self.win_length = 320  # 20 ms
        self.channels = final_channels
        INITIAL_CHANNELS = 64
        cnn_layers = []
        fdim = 161
        for i in range(num_layers):
            num_in_channels = 2 if i == 0 else prev_channels
            curr_channels = min(INITIAL_CHANNELS * 2 ** max(i - 1, 0), final_channels)
            t_stride = 2 if i == num_layers - 1 else 1
            f_stride = 2 if i >= 1 else 1
            cnn_layers.append(
                nn.Conv2d(
                    num_in_channels,
                    curr_channels,
                    (3, 3),
                    stride=(f_stride, t_stride),
                    padding=(0, 1),
                )
            )
            fdim = (fdim - 3) // f_stride + 1
            cnn_layers.append(nn.LeakyReLU(0.1))
            prev_channels = curr_channels
        self.cnn = nn.Sequential(*cnn_layers)
        self.fdim_cnn_out = fdim
        assert (
            curr_channels == final_channels
        ), f"final_channels={final_channels} but last_channels={curr_channels}, choose a different configuration"

    def forward(self, xin):
        compressed = complex_compressed(xin, self.hop_length, self.win_length).permute(
            0, 3, 1, 2
        )
        x = self.cnn(compressed)
        x = torch.flatten(x, 1, -2)
        return x


class ConvTransformerSQAModel(nn.Module):
    def __init__(self, load_weights=True):
        super().__init__()
        self.cnn = ComplexSpecCNN(CNN_FINAL_CHANNELS, N_LAYERS_CNN)
        fdim = self.cnn.fdim_cnn_out
        transformer_pool_attn_conf = Config(
            "",
            None,
            feat_seq_len=FEAT_SEQ_LEN,
            dim_transformer=DIM_TRANSFORMER,
            xlsr_name=None,
            nhead_transformer=N_HEADS,
            nlayers_transformer=N_LAYERS_TRANSFORMER,
        )
        # override the input dimension
        transformer_pool_attn_conf.dim_input = CNN_FINAL_CHANNELS * fdim
        self.transformer_pool_attn = SingleLayerModel(transformer_pool_attn_conf)
        if load_weights:
            chkpt = torch.load(
                DEFAULT_WEIGHTS_CHKPT, map_location="cpu", weights_only=True
            )
            self.load_state_dict(chkpt["model"])
            print("DistillMOS variant 7 weights loaded from:", DEFAULT_WEIGHTS_CHKPT)

    def forward(self, x):
        # segmenting / padding
        wav_overlength = x.shape[1] - SEQ_LEN
        if wav_overlength < 0:
            x = F.pad(x, (0, -wav_overlength))
            wav_overlength = 0
        num_hops = int(np.ceil(wav_overlength / MAX_HOP_LEN)) + 1
        rel_crop_region_start = list(np.linspace(0, 1, num_hops))
        wav_all = [
            x[:, int(c * wav_overlength) : int(c * wav_overlength) + SEQ_LEN]
            for c in rel_crop_region_start
        ]
        x = torch.cat(wav_all, dim=0)  # concatenate along batch dimension

        # normalization
        x = x / (torch.max(torch.abs(x), dim=1, keepdim=True).values + 1e-8)

        # NN processing
        feat = self.cnn(x)
        feat = feat.permute(0, 2, 1)
        norm_qual = self.transformer_pool_attn(feat)

        # average logits over hops
        logit_qual = torch.logit(norm_qual)
        logit_qual = logit_qual.reshape(num_hops, -1, 1)
        logit_qual = torch.mean(logit_qual, dim=0)
        qual = 1 + 4 * torch.sigmoid(logit_qual)
        return qual
