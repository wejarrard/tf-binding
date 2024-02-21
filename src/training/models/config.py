from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EnformerConfig:
    dim: int = 1536
    depth: int = 11
    heads: int = 8
    output_heads: Dict[str, int] = field(
        default_factory=lambda: {"human": 5313, "mouse": 1643}
    )
    target_length: int = 896
    num_cell_lines: int = 33
    attn_dim_key: int = 64
    dropout_rate: float = 0.4
    attn_dropout: float = 0.05
    pos_dropout: float = 0.01
    use_checkpointing: bool = False
    use_convnext: bool = False
    num_downsamples: int = (
        7  # genetic sequence is downsampled 2 ** 7 == 128x in default Enformer
    )
    dim_divisible_by: int = 128
    use_tf_gamma: bool = False
