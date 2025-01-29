from enum import Enum
from typing import Dict, List


class Input(Enum):
    MFCC = 0
    XLSR = 1


class Config:

    name: str = None
    input: Input = None
    feat_seq_len: int = None
    dim_input: int = None
    dim_transformer: int = None
    dim_head_in: int = None
    dim_head_out: int = None

    def __init__(
        self,
        name: str,
        input: Input,
        feat_seq_len: int,
        dim_transformer: int = None,
        xlsr_name: str = None,
        nhead_transformer: int = 4,
        nlayers_transformer: int = 2,
    ):
        if input == Input.MFCC:
            xlsr_name = None

        # Check valid parameters.
        assert feat_seq_len > 0, "feat_seq_len must be positive."
        if xlsr_name is not None:
            _names = ["wav2vec2-xls-r-" + x for x in ["300m", "1b", "2b"]]
            assert xlsr_name in _names, f"xlsr_name must be in {_names}"

        # Save parameters.
        self.name = name
        self.input = input
        self.feat_seq_len = feat_seq_len
        self.dim_transformer = dim_transformer
        self.xlsr_name = xlsr_name
        self.nhead_transformer = nhead_transformer
        self.nlayers_transformer = nlayers_transformer

        if xlsr_name is not None:
            # From XLS-R paper Table 2: Model architectures.
            if xlsr_name == "wav2vec2-xls-r-300m":
                _b = 24
                _h = 1024
            elif xlsr_name == "wav2vec2-xls-r-1b":
                _b = 48
                _h = 1280
            elif xlsr_name == "wav2vec2-xls-r-2b":
                _b = 48
                _h = 1920
            self.xlsr_layers = _b + 1  # +1 for CNN activation "layer0"
            self.dim_input = _h
        else:
            self.xlsr_layers = None
            self.dim_input = 40  # MFCC

        self.dim_head_in = self.dim_transformer  # * self.feat_seq_len
        self.dim_head_out = 1

        self.dropout = 0.0  # TODO: THIS SHOULD HAVE BEEN 0.1!!!!!!


# Length of feature frame window.
FEAT_SEQ_LEN = 384


####################### TRANSFORMER_32DEEP_CONFIG ####################
MFCC_TRANSFORMER_32DEEP_CONFIG = Config(
    "MFCC_TRANSFORMER_32DEEP_CONFIG",
    Input.MFCC,
    feat_seq_len=FEAT_SEQ_LEN,
    dim_transformer=32,
    xlsr_name=None,
    nhead_transformer=4,
    nlayers_transformer=4,
)

XLSR_300M_TRANSFORMER_32DEEP_CONFIG = Config(
    "XLSR_300M_TRANSFORMER_32DEEP_CONFIG",
    Input.XLSR,
    feat_seq_len=FEAT_SEQ_LEN,
    dim_transformer=32,
    xlsr_name="wav2vec2-xls-r-300m",
    nhead_transformer=4,
    nlayers_transformer=4,
)

XLSR_1B_TRANSFORMER_32DEEP_CONFIG = Config(
    "XLSR_1B_TRANSFORMER_32DEEP_CONFIG",
    Input.XLSR,
    feat_seq_len=FEAT_SEQ_LEN,
    dim_transformer=32,
    xlsr_name="wav2vec2-xls-r-1b",
    nhead_transformer=4,
    nlayers_transformer=4,
)

XLSR_2B_TRANSFORMER_32DEEP_CONFIG = Config(
    "XLSR_2B_TRANSFORMER_32DEEP_CONFIG",
    Input.XLSR,
    feat_seq_len=FEAT_SEQ_LEN,
    dim_transformer=32,
    xlsr_name="wav2vec2-xls-r-2b",
    nhead_transformer=4,
    nlayers_transformer=4,
)
