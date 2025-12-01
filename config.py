import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:

    model_name: str = "Qwen/Qwen2.5-0.5B"
    
    dataset_name: str = "glue"
    dataset_config: str = "qnli"
    max_length: int = 200
    train_samples: Optional[int] = None  # None = все данные
    
    # Слои для извлечения
    main_layer: int = 13
    extra_layers: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 14, 15, 16])
    
    num_heads: int = 4
    num_transformer_layers: int = 1
    ff_dim: int = 256
    dropout: float = 0.4
    
    batch_size_extract: int = 32
    batch_size_train: int = 64
    epochs: int = 50
    learning_rate: float = 5e-5
    weight_decay: float = 0.2
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.15
    patience: int = 10
    
    hidden_dropout: float = 0.15
    noise_std: float = 0.02
    mixup_alpha: float = 0.2
    
    save_dir: str = "outputs"

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        assert 0 < self.dropout < 1, "Dropout должен быть в (0, 1)"
        assert self.main_layer > 0, "main_layer должен быть > 0"


def get_config(**kwargs) -> Config:
    return Config(**kwargs)