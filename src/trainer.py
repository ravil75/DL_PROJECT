"""
Trainer для обучения probe
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, Optional

from .losses import LabelSmoothingLoss, compute_kl_loss


class Trainer:
    """
    Trainer с поддержкой:
    - Label smoothing
    - R-Drop (consistency regularization)
    - Mixup augmentation
    - Cosine annealing with warmup
    - Early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98)
        )
        
        # Scheduler
        num_training_steps = len(train_loader) * config.epochs
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._get_lr_lambda(num_warmup_steps, num_training_steps)
        )
        
        # Loss
        self.criterion = LabelSmoothingLoss(smoothing=config.label_smoothing)
        
        # History
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        # Best model tracking
        self.best_acc = 0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Save directory
        os.makedirs(config.save_dir, exist_ok=True)
        self.best_model_path = os.path.join(config.save_dir, 'best_model.pt')
    
    def _get_lr_lambda(self, num_warmup_steps: int, num_training_steps: int):
        """Learning rate schedule: warmup + cosine decay"""
        def lr_lambda(step):
            if step < num_warmup_steps:
                return float(step) / float(max(1, num_warmup_steps))
            progress = float(step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return lr_lambda
    
    def _train_epoch(self) -> float:
        """Одна эпоха обучения"""
        self.model.train()
        total_loss = 0
        num_steps = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for main_seq, main_mask, extra_pooled, labels in pbar:
            main_seq = main_seq.to(self.device)
            main_mask = main_mask.to(self.device)
            extra_pooled = [e.to(self.device) for e in extra_pooled]
            labels = labels.to(self.device)
            
            # Mixup augmentation (50% вероятность)
            use_mixup = self.config.mixup_alpha > 0 and np.random.random() > 0.5
            
            if use_mixup:
                loss = self._mixup_step(main_seq, main_mask, extra_pooled, labels)
            else:
                loss = self._rdrop_step(main_seq, main_mask, extra_pooled, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_steps += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        return total_loss / num_steps
    
    def _mixup_step(self, main_seq, main_mask, extra_pooled, labels) -> torch.Tensor:
        """Шаг с Mixup augmentation"""
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        batch_size = main_seq.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_main = lam * main_seq + (1 - lam) * main_seq[index]
        mixed_extra = [lam * e + (1 - lam) * e[index] for e in extra_pooled]
        
        logits = self.model(mixed_main, main_mask, mixed_extra, add_noise=True)
        loss = lam * self.criterion(logits, labels) + \
               (1 - lam) * self.criterion(logits, labels[index])
        
        return loss
    
    def _rdrop_step(self, main_seq, main_mask, extra_pooled, labels) -> torch.Tensor:
        """Шаг с R-Drop регуляризацией"""
        logits1 = self.model(main_seq, main_mask, extra_pooled, add_noise=True)
        logits2 = self.model(main_seq, main_mask, extra_pooled, add_noise=True)
        
        ce_loss = (self.criterion(logits1, labels) + self.criterion(logits2, labels)) / 2
        kl_loss = compute_kl_loss(logits1, logits2)
        
        return ce_loss + 0.5 * kl_loss
    
    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        """Валидация"""
        self.model.eval()
        total_loss = 0
        num_steps = 0
        all_preds = []
        all_labels = []
        
        for main_seq, main_mask, extra_pooled, labels in self.val_loader:
            main_seq = main_seq.to(self.device)
            main_mask = main_mask.to(self.device)
            extra_pooled = [e.to(self.device) for e in extra_pooled]
            labels_device = labels.to(self.device)
            
            logits = self.model(main_seq, main_mask, extra_pooled, add_noise=False)
            loss = F.cross_entropy(logits, labels_device)
            
            total_loss += loss.item()
            num_steps += 1
            
            preds = logits.argmax(dim=-1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
        
        avg_loss = total_loss / num_steps
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def train(self) -> Tuple[float, Dict]:
        """Основной цикл обучения"""
        print("\n" + "=" * 70)
        print("ROBUST TRAINING")
        print("=" * 70)
        print(f"Model parameters: {self.model.num_parameters:,}")
        print(f"Dropout: {self.config.dropout}")
        print(f"Weight decay: {self.config.weight_decay}")
        print(f"Label smoothing: {self.config.label_smoothing}")
        print(f"Mixup alpha: {self.config.mixup_alpha}")
        print("-" * 70)
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss = self._train_epoch()
            
            # Validate
            val_loss, val_acc = self._validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Check best
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'best_acc': self.best_acc,
                }, self.best_model_path)
                
                marker = "🏆 NEW BEST!"
            else:
                self.patience_counter += 1
                marker = f"(patience: {self.patience_counter}/{self.config.patience})"
            
            print(
                f"Epoch {epoch+1:2d}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
                f"Acc={val_acc:.4f} ({val_acc*100:.2f}%), Best={self.best_acc:.4f} {marker}"
            )
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break
        
        print("-" * 70)
        print(f"✅ Best: Epoch {self.best_epoch}, Acc={self.best_acc:.4f} ({self.best_acc*100:.2f}%)")
        
        # Load best model
        self.load_best()
        
        return self.best_acc, self.history
    
    def load_best(self) -> None:
        """Загрузка лучшей модели"""
        checkpoint = torch.load(self.best_model_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])