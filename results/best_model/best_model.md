# 🚀 HybridProbe: Multi-Layer Fusion на Qwen2.5-0.5B

## 📊 Описание эксперимента

Обучение улучшенного probe-классификатора с использованием hidden states из нескольких слоёв модели Qwen2.5-0.5B. Выбор слоёв основан на предварительном анализе с MLPProbe (слои 13-14 показали лучшие результаты).

**Датасет**: 104 743 train / 5 463 validation (полный QNLI)  
**Модель**: Qwen/Qwen2.5-0.5B (896 hidden dim)  
**Подход**: Transformer Probe + Multi-layer Fusion

## 🏗️ Архитектура

- **Main layer**: 13 (sequences → Transformer encoder)
- **Extra layers**: [9, 10, 11, 12, 14, 15, 16] (pooled → weighted fusion)
- **Learnable layer weights**: автоматический подбор важности слоёв
- **Attention pooling**: агрегация sequence → vector

### Регуляризация
| Метод | Значение |
|-------|----------|
| Dropout | 0.4 |
| Weight Decay | 0.2 |
| Label Smoothing | 0.15 |
| Mixup Alpha | 0.2 |
| R-Drop | ✓ |

## 🎯 Результаты

| Выборка | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| **Train** | 91.61% | 0.9161 | 0.9161 | 0.9161 |
| **Validation** | **91.36%** | 0.9137 | 0.9137 | 0.9136 |

**Overfitting gap**: 0.25% ✅ (минимальный)

### Per-Class Performance (Validation)

| Класс | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Not Entailment | 90.52% | 92.19% | 91.35% | 2 702 |
| Entailment | 92.22% | 90.55% | 91.37% | 2 761 |

### Confidence Analysis
- Mean confidence (correct): **85.70%**
- Mean confidence (wrong): **71.67%**
- High-confidence errors (>0.9): **12**

## 📈 Сравнение с базовыми подходами

| Метод | Val Accuracy | Improvement |
|-------|--------------|-------------|
| LLM Zero-Shot | ~58% | — |
| MLPProbe (layer 14) | 83.93% | — |
| **HybridProbe (multi-layer)** | **91.36%** | **+7.43%** |

## 💡 Выводы

1. **Multi-layer fusion** даёт значительное улучшение над single-layer probe (+7.4%)
2. **Сильная регуляризация** (dropout 0.4, R-Drop, Mixup) предотвращает overfitting
3. **Balanced performance**: обе классы предсказываются одинаково хорошо
4. **Хорошая калибровка**: модель уверена в правильных ответах, менее уверена в ошибках

## 🔧 Гиперпараметры

```python
Config:
    main_layer = 13
    extra_layers = [9, 10, 11, 12, 14, 15, 16]
    num_heads = 4
    num_transformer_layers = 1
    ff_dim = 256
    dropout = 0.4
    learning_rate = 5e-5
    weight_decay = 0.2
    batch_size = 64
    epochs = 50 (early stopping)