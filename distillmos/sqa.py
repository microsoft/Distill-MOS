import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from xls_r_sqa.config import Config, FEAT_SEQ_LEN
from xls_r_sqa.sqa_model import SingleLayerModel

N_LAYERS_CNN = 6
CNN_FINAL_CHANNELS = 256
N_HEADS = 8
DIM_TRANSFORMER = 200
N_LAYERS_TRANSFORMER = 7
SEQ_LEN = 122880
MAX_HOP_LEN = 16000

thispath = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS_CHKPT = os.path.join(thispath, "weights", "distill_mos_v7.pt")


def _complex_compressed(x, hop_length, win_length):
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


class _ComplexSpecCNN(nn.Module):
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
        compressed = _complex_compressed(xin, self.hop_length, self.win_length).permute(
            0, 3, 1, 2
        )
        x = self.cnn(compressed)
        x = torch.flatten(x, 1, -2)
        return x


class ConvTransformerSQAModel(nn.Module):
    """
    Convolutional Neural Network + Transformer model for speech quality assessment.
    """

    def __init__(self, load_weights=True, segmenting_in_forward=True):
        """
        @param load_weights: whether to load the weights from the default checkpoint, default is True
        @param segmenting_in_forward: whether to segment and pad the input in the forward pass, default is True. 
        If False (the preferred option for training/fine-tuning), the input must have shape (batch, 122880)
        """
        super().__init__()
        self.cnn = _ComplexSpecCNN(CNN_FINAL_CHANNELS, N_LAYERS_CNN)
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
        self.segmenting_in_forward = segmenting_in_forward

        if load_weights:
            chkpt = torch.load(DEFAULT_WEIGHTS_CHKPT, map_location="cpu")
            self.load_state_dict(chkpt["model"])
            print("DistillMOS variant 7 weights loaded from:", DEFAULT_WEIGHTS_CHKPT)

    def forward(self, x):
        """
        @param x: 16kHz input waveform, shape (batch, seq_len),
        where seq_len can be arbitrary if segmenting_in_forward is True, and must be 122880 if segmenting_in_forward is False

        @return: predicted MOS score, range 1..5, shape (batch,)
        """
        # segmenting / padding
        if not self.segmenting_in_forward:
            assert (
                x.shape[1] == SEQ_LEN
            ), f"input shape {x.shape} does not match SEQ_LEN={SEQ_LEN}"
        else:
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

        if self.segmenting_in_forward:
            norm_qual = norm_qual.reshape(num_hops, -1, 1)
            norm_qual = torch.mean(norm_qual, dim=0)

        return 1 + 4 * norm_qual
