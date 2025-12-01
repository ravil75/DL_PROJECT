"""
Скрипт обучения QNLI Probe
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import warnings
warnings.filterwarnings('ignore')

from config import Config
from src.data import (
    load_model_and_tokenizer,
    load_data,
    format_input,
    extract_hybrid_data,
    HybridDataset,
    create_dataloaders,
    cleanup_llm
)
from src.model import HybridProbe 
from src.trainer import Trainer
from src.utils import set_seed, check_environment


def main():
    # Проверка окружения
    check_environment()
    
    # Конфигурация
    config = Config()
    set_seed(42)
    
    # Загрузка LLM
    model, tokenizer = load_model_and_tokenizer(config.model_name)
    hidden_dim = model.config.hidden_size
    
    # Загрузка данных
    train_data, val_data = load_data(
        config.dataset_name,
        config.dataset_config,
        config.train_samples
    )
    
    # Подготовка текстов
    print("\nПодготовка текстов...")
    texts_train = [format_input(ex) for ex in train_data]
    texts_val = [format_input(ex) for ex in val_data]
    labels_train = torch.tensor([ex["label"] for ex in train_data])
    labels_val = torch.tensor([ex["label"] for ex in val_data])
    
    print(f"\n✅ Данные готовы")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   Train texts: {len(texts_train)}")
    print(f"   Val texts: {len(texts_val)}")
    
    # Извлечение hidden states
    print("\n" + "=" * 60)
    print("ИЗВЛЕЧЕНИЕ HIDDEN STATES")
    print("=" * 60)
    
    main_hidden_train, main_masks_train, extra_pooled_train = extract_hybrid_data(
        model, tokenizer, texts_train,
        config.main_layer, config.extra_layers,
        config.max_length, config.batch_size_extract, config.device
    )
    
    main_hidden_val, main_masks_val, extra_pooled_val = extract_hybrid_data(
        model, tokenizer, texts_val,
        config.main_layer, config.extra_layers,
        config.max_length, config.batch_size_extract, config.device
    )
    
    # Освобождение памяти от LLM
    cleanup_llm(model)
    
    # Создание датасетов
    train_dataset = HybridDataset(
        main_hidden_train, main_masks_train,
        extra_pooled_train, labels_train, config.extra_layers
    )
    val_dataset = HybridDataset(
        main_hidden_val, main_masks_val,
        extra_pooled_val, labels_val, config.extra_layers
    )
    
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, config.batch_size_train
    )
    
    print(f"✅ Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"✅ Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # Создание модели
    probe = HybridProbe(
        hidden_dim=hidden_dim,
        num_extra_layers=len(config.extra_layers),
        num_heads=config.num_heads,
        num_transformer_layers=config.num_transformer_layers,
        ff_dim=config.ff_dim,
        max_seq_len=config.max_length,
        dropout=config.dropout,
        noise_std=config.noise_std
    )
    
    print(f"\n🔧 Model: {probe.num_parameters:,} parameters")
    
    # Обучение
    trainer = Trainer(probe, train_loader, val_loader, config, config.device)
    best_acc, history = trainer.train()
    
    print(f"\n🎉 Обучение завершено! Best accuracy: {best_acc:.4f}")
    
    return best_acc, history, trainer


if __name__ == "__main__":
    main()