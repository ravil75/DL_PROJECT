import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import torch
import time
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config
from src.model import HybridProbe


# ============================================================================
#                              PYDANTIC MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Запрос на предсказание"""
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=500,
        description="Вопрос на английском языке",
        examples=["What is the capital of France?"]
    )
    sentence: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="Предложение для проверки",
        examples=["Paris is the capital and most populous city of France."]
    )


class PredictionResponse(BaseModel):
    """Ответ с предсказанием"""
    prediction: int = Field(..., description="Класс: 0=entailment, 1=not_entailment")
    label: str = Field(..., description="Текстовая метка класса")
    confidence: float = Field(..., ge=0, le=1, description="Уверенность модели")
    prob_entailment: float = Field(..., ge=0, le=1, description="P(entailment)")
    prob_not_entailment: float = Field(..., ge=0, le=1, description="P(not_entailment)")
    inference_time_ms: float = Field(..., description="Время инференса в миллисекундах")


class BatchRequest(BaseModel):
    """Batch запрос"""
    items: List[PredictionRequest] = Field(
        ..., 
        max_length=32,
        description="Список пар вопрос-предложение (максимум 32)"
    )


class BatchResponse(BaseModel):
    """Batch ответ"""
    results: List[PredictionResponse]
    total_time_ms: float
    count: int


class HealthResponse(BaseModel):
    """Статус сервиса"""
    status: str
    model_loaded: bool
    model_name: str
    probe_parameters: int
    best_accuracy: Optional[float]
    device: str


class ExampleItem(BaseModel):
    """Пример для тестирования"""
    question: str
    sentence: str
    expected: str


class ExamplesResponse(BaseModel):
    """Список примеров"""
    examples: List[ExampleItem]


# ============================================================================
#                              GLOBAL STATE
# ============================================================================

class ModelState:
    """Глобальное состояние с моделями"""
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.probe = None
        self.config = None
        self.best_acc = None
        self.is_loaded = False
        self.device = None


state = ModelState()


# ============================================================================
#                              загрузка моделей
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Загрузка моделей при старте сервера.
    Освобождение ресурсов при остановке.
    """
    print("=" * 60)
    print("ЗАПУСК FASTAPI СЕРВЕРА")
    print("=" * 60)
    
    state.config = Config()
    state.device = state.config.device
    print(f"Device: {state.device}")
    
    # Проверка наличия модели
    checkpoint_path = os.path.join(PROJECT_ROOT, state.config.save_dir, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Модель не найдена: {checkpoint_path}")
        print("   Сначала обучите модель: python scripts/train.py")
        # Продолжаем работу, но модель не будет загружена
        yield
        return

    print(f"\n Загрузка токенизатора: {state.config.model_name}")
    state.tokenizer = AutoTokenizer.from_pretrained(state.config.model_name)
    if state.tokenizer.pad_token is None:
        state.tokenizer.pad_token = state.tokenizer.eos_token
    print("   ✅ Токенизатор загружен")
    
    # Загрузка LLM
    print(f"\n Загрузка LLM: {state.config.model_name}")
    state.llm = AutoModelForCausalLM.from_pretrained(
        state.config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True
    )
    state.llm.eval()
    hidden_dim = state.llm.config.hidden_size
    print(f"   ✅ LLM загружен (hidden_dim={hidden_dim})")
    
    # Создание Probe
    print(f"\n Создание Probe модели")
    state.probe = HybridProbe(
        hidden_dim=hidden_dim,
        num_extra_layers=len(state.config.extra_layers),
        num_heads=state.config.num_heads,
        num_transformer_layers=state.config.num_transformer_layers,
        ff_dim=state.config.ff_dim,
        max_seq_len=state.config.max_length,
        dropout=state.config.dropout,
        noise_std=state.config.noise_std
    )
    
    # Загрузка весов Probe
    print(f"\n Загрузка весов: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=state.device, weights_only=False)
    state.probe.load_state_dict(checkpoint['model_state_dict'])
    state.best_acc = checkpoint.get('best_acc', None)
    print(f"   ✅ Веса загружены (best_acc={state.best_acc:.4f})" if state.best_acc else "   ✅ Веса загружены")
    
    state.probe = state.probe.to(state.device)
    state.probe.eval()
    state.is_loaded = True
    
    print("\n" + "=" * 60)
    print(f"✅ СЕРВЕР ГОТОВ")
    print(f"   Probe parameters: {state.probe.num_parameters:,}")
    print(f"   API docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    
    yield
    
    # Освобождение ресурсов
    print("\n Освобождение ресурсов...")
    del state.llm
    del state.tokenizer
    del state.probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ Ресурсы освобождены")


# ============================================================================
#                              FASTAPI APP
# ============================================================================

app = FastAPI(
    title="QNLI Probe API",
    description="""
    ## API для классификации Question-Answering NLI
    
    Определяет, содержит ли предложение ответ на заданный вопрос.
    
    ### Классы:
    - **entailment (0)**: Предложение содержит ответ
    - **not_entailment (1)**: Предложение НЕ содержит ответ
    
    ### Endpoints:
    - `POST /predict` - одиночное предсказание
    - `POST /predict/batch` - batch предсказание (до 32 примеров)
    - `GET /health` - статус сервиса
    - `GET /examples` - примеры для тестирования
    """,
    version="1.0.0",
    lifespan=lifespan
)

# разрешаем запросы с любых доменов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
#                              вспомогательные функции
# ============================================================================

def format_input(question: str, sentence: str) -> str:
    """Форматирование входа (как при обучении)"""
    return (
        f"QNLI Task.\n"
        f"Does the sentence \"{sentence}\" contain the answer "
        f"to the question \"{question}\"?\n"
        f"Answer:"
    )


@torch.no_grad()
def run_inference(question: str, sentence: str) -> dict:
    """
    Выполнение инференса для одной пары вопрос-предложение.
    
    Returns:
        dict с prediction, label, confidence, probabilities
    """
    # Форматируем вход
    text = format_input(question, sentence)
    
    # Токенизация
    inputs = state.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=state.config.max_length
    ).to(state.device)
    
    # Получаем hidden states из LLM
    outputs = state.llm(**inputs, output_hidden_states=True)
    attention_mask = inputs["attention_mask"]
    seq_length = attention_mask.sum() - 1
    
    # Main layer — полная последовательность
    main_hidden = outputs.hidden_states[state.config.main_layer]
    real_len = int(attention_mask[0].sum().item())
    main_seq = main_hidden[0, :real_len, :].unsqueeze(0).float()
    main_mask = attention_mask[0, :real_len].unsqueeze(0).float()
    
    # Extra layers — pooled (последний токен)
    extra_pooled = []
    for layer in state.config.extra_layers:
        hidden = outputs.hidden_states[layer]
        pooled = hidden[0, seq_length, :].unsqueeze(0).float()
        extra_pooled.append(pooled)
    
    # Предсказание через Probe
    logits = state.probe(main_seq, main_mask, extra_pooled, add_noise=False)
    probs = torch.softmax(logits, dim=-1)
    pred = logits.argmax(dim=-1).item()
    
    return {
        'prediction': pred,
        'label': 'entailment' if pred == 0 else 'not_entailment',
        'confidence': float(probs[0, pred].item()),
        'prob_entailment': float(probs[0, 0].item()),
        'prob_not_entailment': float(probs[0, 1].item())
    }


# ============================================================================
#                              API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Корневой endpoint с информацией об API"""
    return {
        "message": "🔍 QNLI Probe API",
        "description": "API для классификации Question-Answering NLI",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch": "/predict/batch",
            "examples": "/examples"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """
    Проверка состояния сервиса.
    Возвращает информацию о загруженных моделях и их параметрах.
    """
    return HealthResponse(
        status="healthy" if state.is_loaded else "model_not_loaded",
        model_loaded=state.is_loaded,
        model_name=state.config.model_name if state.config else "N/A",
        probe_parameters=state.probe.num_parameters if state.probe else 0,
        best_accuracy=state.best_acc,
        device=str(state.device) if state.device else "N/A"
    )


@app.get("/examples", response_model=ExamplesResponse, tags=["Info"])
async def get_examples():
    """
    Получить примеры для тестирования API.
    Возвращает список пар вопрос-предложение с ожидаемыми ответами.
    """
    examples = [
        ExampleItem(
            question="What is the capital of France?",
            sentence="Paris is the capital and most populous city of France.",
            expected="entailment"
        ),
        ExampleItem(
            question="When was Python created?",
            sentence="Python was conceived in the late 1980s by Guido van Rossum.",
            expected="entailment"
        ),
        ExampleItem(
            question="What is the speed of light?",
            sentence="Einstein developed the theory of relativity.",
            expected="not_entailment"
        ),
        ExampleItem(
            question="How many planets are in the solar system?",
            sentence="The weather today is sunny and warm.",
            expected="not_entailment"
        ),
        ExampleItem(
            question="Who wrote Romeo and Juliet?",
            sentence="William Shakespeare was an English playwright and poet.",
            expected="entailment"
        ),
        ExampleItem(
            question="What is machine learning?",
            sentence="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            expected="entailment"
        )
    ]
    return ExamplesResponse(examples=examples)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Предсказание для одной пары вопрос-предложение.
    
    ### Параметры:
    - **question**: Вопрос на английском языке
    - **sentence**: Предложение, которое может содержать ответ
    
    ### Возвращает:
    - **prediction**: Числовой класс (0 или 1)
    - **label**: Текстовая метка (entailment / not_entailment)
    - **confidence**: Уверенность модели (0-1)
    - **prob_entailment**: Вероятность класса entailment
    - **prob_not_entailment**: Вероятность класса not_entailment
    - **inference_time_ms**: Время инференса в миллисекундах
    """
    # Проверяем, загружена ли модель
    if not state.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена. Проверьте /health для деталей."
        )
    
    try:
        start_time = time.time()
        result = run_inference(request.question, request.sentence)
        inference_time_ms = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            **result,
            inference_time_ms=round(inference_time_ms, 2)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка инференса: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(request: BatchRequest):
    """
    Batch предсказание для нескольких пар вопрос-предложение.
    
    ### Ограничения:
    - Максимум 32 примера за один запрос
    
    ### Параметры:
    - **items**: Список объектов с полями question и sentence
    
    ### Возвращает:
    - **results**: Список предсказаний
    - **total_time_ms**: Общее время обработки
    - **count**: Количество обработанных примеров
    """
    if not state.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена. Проверьте /health для деталей."
        )
    
    start_time = time.time()
    results = []
    
    for item in request.items:
        try:
            item_start = time.time()
            result = run_inference(item.question, item.sentence)
            item_time_ms = (time.time() - item_start) * 1000
            
            results.append(PredictionResponse(
                **result,
                inference_time_ms=round(item_time_ms, 2)
            ))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ошибка при обработке примера: {str(e)}"
            )
    
    total_time_ms = (time.time() - start_time) * 1000
    
    return BatchResponse(
        results=results,
        total_time_ms=round(total_time_ms, 2),
        count=len(results)
    )


# ============================================================================
#                              ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )