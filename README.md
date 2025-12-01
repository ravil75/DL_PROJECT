# QNLI Probe: Extracting Knowledge from LLM Hidden States

Probing classifier для задачи QNLI (Question-answering Natural Language Inference) на основе hidden states из LLM.

## 🎯 Описание

Проект извлекает внутренние представления (hidden states) из языковой модели Qwen2.5-0.5B и обучает легковесный probe-классификатор для задачи QNLI.

### Ключевые особенности:
- **Multi-layer fusion**: комбинация hidden states из нескольких слоёв
- **Attention pooling**: умное агрегирование sequence → vector
- **Robust training**: R-Drop, Mixup, Label Smoothing

## 📁 Структура