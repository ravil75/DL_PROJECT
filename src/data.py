"""
Загрузка данных и извлечение hidden states
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
import gc


def format_input(example: dict) -> str:
    """Форматирование примера QNLI в промпт"""
    question = example['question']
    sentence = example['sentence']
    return (
        f"QNLI Task.\n"
        f"Does the sentence \"{sentence}\" contain the answer "
        f"to the question \"{question}\"?\n"
        f"Answer:"
    )


def load_model_and_tokenizer(model_name: str) -> Tuple:
    """Загрузка LLM для извлечения hidden states"""
    print(f"\n{'='*60}")
    print("ЗАГРУЗКА МОДЕЛИ")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True
    )
    model.eval()
    
    print(f"✅ Модель загружена: {model_name}")
    print(f"   Hidden size: {model.config.hidden_size}")
    print(f"   Num layers: {model.config.num_hidden_layers}")
    
    return model, tokenizer


def load_data(
    dataset_name: str,
    dataset_config: str,
    train_samples: Optional[int] = None
) -> Tuple:
    """Загрузка QNLI датасета"""
    print(f"\n{'='*60}")
    print("ЗАГРУЗКА ДАННЫХ")
    print("="*60)
    
    dataset = load_dataset(dataset_name, dataset_config)
    
    if train_samples is None:
        train_data = dataset["train"]
    else:
        train_data = dataset["train"].select(
            range(min(train_samples, len(dataset["train"])))
        )
    
    val_data = dataset["validation"]
    
    print(f"✅ Train: {len(train_data)} samples")
    print(f"✅ Val: {len(val_data)} samples")
    
    # Баланс классов
    train_labels = [ex["label"] for ex in train_data]
    val_labels = [ex["label"] for ex in val_data]
    
    print(f"\n📊 Train баланс: class0={train_labels.count(0)}, class1={train_labels.count(1)}")
    print(f"📊 Val баланс: class0={val_labels.count(0)}, class1={val_labels.count(1)}")
    
    return train_data, val_data


def extract_hybrid_data(
    model,
    tokenizer,
    texts: List[str],
    main_layer: int,
    extra_layers: List[int],
    max_length: int,
    batch_size: int,
    device: str
) -> Tuple[List, List, Dict]:
    """
    Извлечение hidden states из LLM.
    
    Returns:
        main_hidden: список тензоров (seq_len, hidden_dim) для main_layer
        main_masks: список масок
        extra_pooled: dict {layer: tensor (N, hidden_dim)}
    """
    print(f"\n📤 Извлекаем hidden states...")
    print(f"   Main layer {main_layer}: sequences")
    print(f"   Extra layers {extra_layers}: pooled")
    
    main_hidden = []
    main_masks = []
    extra_pooled = {layer: [] for layer in extra_layers}
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
            batch_texts = texts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)
            
            outputs = model(**inputs, output_hidden_states=True)
            attention_mask = inputs["attention_mask"]
            
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(attention_mask.size(0), device=device)
            
            # Main layer sequences
            main = outputs.hidden_states[main_layer]
            
            for j in range(main.size(0)):
                real_len = int(attention_mask[j].sum().item())
                hidden_fp16 = main[j, :real_len, :].cpu().half()
                
                # Проверка на NaN/Inf
                if torch.isnan(hidden_fp16).any() or torch.isinf(hidden_fp16).any():
                    print(f"⚠️ NaN/Inf в примере {i+j}, используем float32")
                    hidden_fp16 = main[j, :real_len, :].cpu().float()
                
                main_hidden.append(hidden_fp16)
                main_masks.append(attention_mask[j, :real_len].cpu().half())
            
            # Extra layers pooled (last token)
            for layer in extra_layers:
                hidden = outputs.hidden_states[layer]
                pooled = hidden[batch_indices, seq_lengths]
                extra_pooled[layer].append(pooled.cpu().half())
            
            del outputs, inputs
            torch.cuda.empty_cache()
    
    # Конкатенация extra_pooled
    for layer in extra_layers:
        extra_pooled[layer] = torch.cat(extra_pooled[layer], dim=0)
    
    return main_hidden, main_masks, extra_pooled


class HybridDataset(Dataset):
    """
    Dataset для probe.
    Хранит данные в float16, конвертирует в float32 при доступе.
    """
    
    def __init__(
        self,
        main_hidden: List[torch.Tensor],
        main_masks: List[torch.Tensor],
        extra_pooled_dict: Dict[int, torch.Tensor],
        labels: torch.Tensor,
        extra_layers: List[int]
    ):
        self.main_hidden = main_hidden
        self.main_masks = main_masks
        self.extra_layers = extra_layers
        self.extra_pooled = [extra_pooled_dict[l] for l in extra_layers]
        self.labels = labels
        
        print(f"HybridDataset: {len(labels)} samples")
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int):
        main_seq = self.main_hidden[idx].float()
        main_mask = self.main_masks[idx].float()
        extra = [self.extra_pooled[i][idx].float() for i in range(len(self.extra_layers))]
        label = self.labels[idx]
        
        return main_seq, main_mask, extra, label


def collate_hybrid(batch: List) -> Tuple:
    """Collate function с динамическим padding"""
    batch_size = len(batch)
    num_extra = len(batch[0][2])
    
    max_len = max(b[0].size(0) for b in batch)
    hidden_dim = batch[0][0].size(1)
    
    main_padded = torch.zeros(batch_size, max_len, hidden_dim)
    mask_padded = torch.zeros(batch_size, max_len)
    
    for i, b in enumerate(batch):
        seq_len = b[0].size(0)
        main_padded[i, :seq_len, :] = b[0]
        mask_padded[i, :seq_len] = b[1]
    
    extra_pooled = [torch.stack([b[2][j] for b in batch]) for j in range(num_extra)]
    labels = torch.stack([b[3] for b in batch])
    
    return main_padded, mask_padded, extra_pooled, labels


def create_dataloaders(
    train_dataset: HybridDataset,
    val_dataset: HybridDataset,
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Создание DataLoader'ов"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_hybrid,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_hybrid,
        pin_memory=True
    )
    
    return train_loader, val_loader


def cleanup_llm(model) -> None:
    """Освобождение памяти от LLM"""
    print("\n🧹 Освобождаем память от LLM...")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Память освобождена")