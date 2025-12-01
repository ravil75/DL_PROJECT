"""
Вспомогательные функции
"""
import torch
import numpy as np
import random
from typing import Tuple


def set_seed(seed: int = 42) -> None:
    """Фиксация seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mixup_data(
    x1: torch.Tensor,
    x2: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    Mixup augmentation на hidden states.
    
    Mixup создаёт виртуальные примеры путём линейной интерполяции.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x1.size(0)
    index = torch.randperm(batch_size).to(x1.device)
    
    mixed_x1 = lam * x1 + (1 - lam) * x1[index]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index]
    y_a, y_b = y, y[index]
    
    return mixed_x1, mixed_x2, y_a, y_b, lam, index


def check_environment() -> None:
    """Проверка окружения"""
    print("=" * 60)
    print("ПРОВЕРКА ОКРУЖЕНИЯ")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Memory: {props.total_memory / 1e9:.1f} GB")