"""
Скрипт оценки и визуализации
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
from typing import Dict


def evaluate_model(model, val_loader, device: str = 'cuda') -> Dict:
    """Полная оценка модели"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for main_seq, main_mask, extra_pooled, labels in tqdm(val_loader, desc="Evaluating"):
            main_seq = main_seq.to(device)
            main_mask = main_mask.to(device)
            extra_pooled = [e.to(device) for e in extra_pooled]
            
            logits = model(main_seq, main_mask, extra_pooled, add_noise=False)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    return {
        'accuracy': acc,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }


def plot_training(history: Dict, save_path: str = 'training_curves.png'):
    """Графики обучения"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Train & Val Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', lw=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', lw=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Train & Val Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Val Accuracy
    axes[1].plot(epochs, history['val_acc'], 'g-', lw=2, marker='o', ms=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    best_acc = max(history['val_acc'])
    best_epoch = np.argmax(history['val_acc']) + 1
    axes[1].set_title(f'Val Accuracy (Best: {best_acc:.4f})')
    axes[1].axhline(y=best_acc, color='red', linestyle='--', alpha=0.5)
    axes[1].scatter([best_epoch], [best_acc], color='red', s=100, zorder=5)
    axes[1].grid(True, alpha=0.3)
    
    # Overfitting Gap
    gap = np.array(history['val_loss']) - np.array(history['train_loss'])
    colors = ['red' if g > 0 else 'green' for g in gap]
    axes[2].bar(epochs, gap, color=colors, alpha=0.7)
    axes[2].axhline(y=0, color='black', linestyle='-')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Val Loss - Train Loss')
    axes[2].set_title('Overfitting Gap')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    print(f"\n📊 Saved to {save_path}")


def plot_confusion_matrix(results: Dict, save_path: str = 'confusion_matrix.png'):
    """Confusion matrix"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm = confusion_matrix(results['labels'], results['predictions'])
    
    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title(f'Confusion Matrix (Acc: {results["accuracy"]:.4f})')
    
    # Normalized
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Normalized')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    print("\n📋 CLASSIFICATION REPORT:")
    print(classification_report(
        results['labels'], 
        results['predictions'],
        target_names=['Class 0', 'Class 1']
    ))


def plot_confidence(results: Dict, save_path: str = 'confidence.png'):
    """Анализ уверенности"""
    probs = results['probabilities']
    preds = results['predictions']
    labels = results['labels']
    
    confidence = np.maximum(probs, 1 - probs)
    correct = (preds == labels)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(confidence[correct], bins=25, alpha=0.7, label='Correct', color='green')
    axes[0].hist(confidence[~correct], bins=25, alpha=0.7, label='Wrong', color='red')
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Confidence Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    stats = f"""
    Correct: {correct.sum()} ({correct.mean()*100:.1f}%)
    Wrong: {(~correct).sum()} ({(~correct).mean()*100:.1f}%)
    
    Mean confidence (correct): {confidence[correct].mean():.3f}
    Mean confidence (wrong): {confidence[~correct].mean():.3f}
    
    High-confidence errors (>0.9): {((confidence > 0.9) & (~correct)).sum()}
    """
    axes[1].text(0.1, 0.5, stats, fontsize=12, family='monospace',
                 transform=axes[1].transAxes, verticalalignment='center')
    axes[1].axis('off')
    axes[1].set_title('Statistics')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    print("Use: from scripts.evaluate import evaluate_model, plot_training, ...")