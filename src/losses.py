"""
Loss функции для обучения
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss с label smoothing.
    
    Label smoothing уменьшает overconfidence модели и улучшает генерализацию.
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))


def compute_kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Симметричный KL divergence для R-Drop регуляризации.
    
    R-Drop: два forward pass через модель с dropout создают разные выходы.
    KL loss заставляет эти выходы быть согласованными.
    """
    p_loss = F.kl_div(
        F.log_softmax(p, dim=-1),
        F.softmax(q, dim=-1),
        reduction='batchmean'
    )
    q_loss = F.kl_div(
        F.log_softmax(q, dim=-1),
        F.softmax(p, dim=-1),
        reduction='batchmean'
    )
    return (p_loss + q_loss) / 2