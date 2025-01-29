from pathlib import Path
import torch
from torchaudio.transforms import MFCC
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from typing import Iterable, Tuple, Union
import warnings

from .center_crop import CenterCrop
from .config import Config, FEAT_SEQ_LEN
from .sqa_model import FusionModel, SingleLayerModel

class E2EModel(torch.nn.Module):
    def __init__(
            self,
            config: Config,
            xlsr_layers: Union[int, Tuple[int,int]],
            xlsr_dir: Path,
            sqa_path: Path,
        ):
        super().__init__()

        # Parse config.
        self.config = config
        self.use_xlsr = "xlsr" in config.name.lower()
        if self.use_xlsr:
            xlsr_name = config.xlsr_name
            xlsr_size = xlsr_name[len("wav2vec2-xls-r-"):]
            assert xlsr_size in ["300m", "1b", "2b"]

        # Parse xlsr_layers.
        if self.use_xlsr:
            if xlsr_layers is None:
                raise Exception("Please specify xlsr_layers!")
            if isinstance(xlsr_layers, int):
                xlsr_layers = (xlsr_layers,)
            if not (isinstance(xlsr_layers, Iterable) and len(xlsr_layers) in [1,2]):
                raise Exception("xlsr_layers must either be one or two integers")
            self.xlsr_layers = xlsr_layers
            self.is_fusion = len(xlsr_layers) == 2

            # Check correct layer input.
            if xlsr_size == "300m":
                _possible_layers = [5,21]
            else:
                _possible_layers = [10,41]
            for _layer in xlsr_layers:
                if _layer not in _possible_layers:
                    raise Exception(f"Invalid layer! Possible layers for XLS-R {xlsr_size}: {_possible_layers}")
        else:
            self.xlsr_layers = None # MFCC
            self.is_fusion = False

        
        # XLS-R / MFCC feature extractor.
        self.sample_rate = 16000
        if self.use_xlsr:
            self.w2v2_feat_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=self.sample_rate,
                padding_value=0.0,
                do_normalize=True,
                return_attention_mask=True
            )
            self.xlsr_model = Wav2Vec2Model.from_pretrained(xlsr_dir)
        else: 
            self.mfcc_calculator = MFCC(sample_rate=self.sample_rate)

        # Center crop.
        self.crop = CenterCrop(FEAT_SEQ_LEN)

        # SQA model.
        if self.is_fusion:
            self.sqa_model = FusionModel(config)
        else:
            self.sqa_model = SingleLayerModel(config)
        self.sqa_model.load_state_dict(torch.load(sqa_path))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        r"""Predict perceived speech quality in [1,5] for the given audio sample.

        Args:
            audio: audio sample to process.

        Shape:
            - features: (N,T) or (T,)
            - output: (N,1) or (1,)
        """

        unsqueezed = False
        if audio.dim() == 1:
            audio = audio.unsqueeze(0) # Include batch dim
            unsqueezed = True

        if audio.size(1) > 15 * self.sample_rate:
            warnings.warn("Warning: audio is longer than 15 seconds. It might be better to compute predictions over a sliding window and take the average.")

        # Calculate XLS-R / MFCC features.
        if self.use_xlsr:
            audio_np = audio.detach().cpu().numpy()
            xlsr_input = self.w2v2_feat_extractor(
                audio_np,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )["input_values"].squeeze(0).to(audio.device)
            with torch.no_grad():
                xlsr_out = self.xlsr_model(xlsr_input, output_hidden_states=True)
                x = [self.crop(xlsr_out.hidden_states[i].squeeze(0)) for i in self.xlsr_layers]
                if len(self.xlsr_layers) == 1:
                    x = x[0] # Extract from list for SingleLayerModel
        else:
            mfcc: torch.Tensor = self.mfcc_calculator(audio)
            mfcc = mfcc.transpose(-1, -2) # transpose from (N, n_mfcc, L) to (N, L, n_mfcc)
            with torch.no_grad():
                x = self.crop(mfcc.to(audio.device))

        # Calculate MOS prediction.
        sqa_out = self.sqa_model(x)
        sqa_out_denorm = 1.0 + sqa_out * 4.0
        if unsqueezed:
            sqa_out_denorm = sqa_out_denorm.squeeze(0)
        return sqa_out_denorm

