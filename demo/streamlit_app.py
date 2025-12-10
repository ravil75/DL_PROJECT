"""
Streamlit Frontend для QNLI Probe API
=====================================

Запуск:
    streamlit run demo/streamlit_app.py

Требует работающий API сервер:
    uvicorn demo.api.main:app --port 8000
"""

import streamlit as st
import requests
import pandas as pd
import time
import os
from typing import Optional, Dict, List

# URL API сервера
DEFAULT_API_URL = "http://localhost:8000"

# Путь к графикам (относительно корня проекта)
FIGURES_PATH = "results/best_model/figures"

# ============================================================================
#                              НАСТРОЙКА СТРАНИЦЫ
# ============================================================================

st.set_page_config(
    page_title="QNLI Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
#                              CSS СТИЛИ
# ============================================================================

st.markdown("""
<style>
    /* Главный заголовок */
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Карточки результатов */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .result-entailment {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
    }
    
    .result-not-entailment {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
    }
    
    .result-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
    }
    
    .result-description {
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Статус API */
    .api-status {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    
    .api-online {
        background-color: #d4edda;
        color: #155724;
    }
    
    .api-offline {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* Метрики */
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    /* Прогресс бары */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Карточки для графиков */
    .figure-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }
    
    .figure-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .figure-description {
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    
    /* Информационные блоки */
    .info-box {
        background: linear-gradient(135deg, #e8f4fd 0%, #d6eaf8 100%);
        border-left: 4px solid #3498db;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f8e8 0%, #d4edda 100%);
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff8e8 0%, #ffeeba 100%);
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
#                              API ФУНКЦИИ
# ============================================================================

def check_api_health(api_url: str) -> Optional[Dict]:
    """Проверка доступности API"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def get_examples(api_url: str) -> List[Dict]:
    """Получение примеров из API"""
    try:
        response = requests.get(f"{api_url}/examples", timeout=5)
        if response.status_code == 200:
            return response.json().get("examples", [])
        return []
    except requests.exceptions.RequestException:
        return []


def predict_single(api_url: str, question: str, sentence: str) -> Optional[Dict]:
    """Одиночное предсказание через API"""
    try:
        response = requests.post(
            f"{api_url}/predict",
            json={"question": question, "sentence": sentence},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return None


def predict_batch(api_url: str, items: List[Dict]) -> Optional[Dict]:
    """Batch предсказание через API"""
    try:
        response = requests.post(
            f"{api_url}/predict/batch",
            json={"items": items},
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return None


# ============================================================================
#                              ФУНКЦИИ ДЛЯ ГРАФИКОВ
# ============================================================================

def get_figures_path() -> str:
    """Получение пути к папке с графиками"""
    # Пробуем разные пути к папке с графиками
    possible_paths = [
        FIGURES_PATH,
        f"/content/DL_PROJECT/{FIGURES_PATH}",
        f"../{FIGURES_PATH}",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), FIGURES_PATH)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return FIGURES_PATH


def check_figures_exist(figures_path: str) -> Dict[str, bool]:
    """Проверка наличия файлов с графиками"""
    figures = {
        "training_curves": os.path.exists(os.path.join(figures_path, "training_curves.png")),
        "confusion_matrix": os.path.exists(os.path.join(figures_path, "confusion_matrix.png")),
        "confidence_analysis": os.path.exists(os.path.join(figures_path, "confidence_analysis.png"))
    }
    return figures


# ============================================================================
#                              UI КОМПОНЕНТЫ
# ============================================================================

def render_header():
    """Отрисовка заголовка"""
    st.markdown(
        '<h1 class="main-title">🔍 QNLI Question Answering Classifier</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">Определяет, содержит ли предложение ответ на заданный вопрос</p>',
        unsafe_allow_html=True
    )


def render_sidebar(api_url: str) -> tuple:
    """Отрисовка боковой панели"""
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # URL API
        new_api_url = st.text_input(
            "API URL:",
            value=api_url,
            help="URL FastAPI сервера"
        )
        
        st.markdown("---")
        
        # Проверка статуса API
        st.subheader("📡 Статус API")
        
        health = check_api_health(new_api_url)
        
        if health:
            st.markdown(
                '<span class="api-status api-online">🟢 Online</span>',
                unsafe_allow_html=True
            )
            
            st.markdown("---")
            st.markdown("**📊 Информация о модели:**")
            st.markdown(f"- **Модель:** `{health['model_name'].split('/')[-1]}`")
            st.markdown(f"- **Параметры:** `{health['probe_parameters']:,}`")
            st.markdown(f"- **Device:** `{health['device']}`")
            
            if health['best_accuracy']:
                st.markdown(f"- **Accuracy:** `{health['best_accuracy']:.2%}`")
            
            model_loaded = health['model_loaded']
        else:
            st.markdown(
                '<span class="api-status api-offline">🔴 Offline</span>',
                unsafe_allow_html=True
            )
            st.error("API недоступен!")
            st.info(
                "Запустите сервер:\n"
                "```\n"
                "uvicorn demo.api.main:app --port 8000\n"
                "```"
            )
            model_loaded = False
        
        st.markdown("---")
        
        # Информация о классах
        st.subheader("🎯 Классы")
        st.markdown("""
        - **Entailment (0):**  
          Предложение содержит ответ
        - **Not Entailment (1):**  
          Предложение НЕ содержит ответ
        """)
        
        return new_api_url, model_loaded


def render_result(result: Dict):
    """Отрисовка результата предсказания"""
    if result['prediction'] == 0:
        st.markdown("""
        <div class="result-card result-entailment">
            <p class="result-title" style="color: #155724;">✅ ENTAILMENT</p>
            <p class="result-description" style="color: #155724;">
                Предложение <strong>СОДЕРЖИТ</strong> ответ на вопрос
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card result-not-entailment">
            <p class="result-title" style="color: #721c24;">❌ NOT ENTAILMENT</p>
            <p class="result-description" style="color: #721c24;">
                Предложение <strong>НЕ СОДЕРЖИТ</strong> ответ на вопрос
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Метрики
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🎯 Уверенность",
            value=f"{result['confidence']:.1%}"
        )
    
    with col2:
        st.metric(
            label="⏱️ Время",
            value=f"{result['inference_time_ms']:.0f} мс"
        )
    
    with col3:
        st.metric(
            label="🏷️ Класс",
            value=result['label']
        )
    
    # Вероятности
    st.markdown("---")
    st.markdown("**📊 Распределение вероятностей:**")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        prob_ent = result['prob_entailment']
        st.markdown(f"**Entailment:** `{prob_ent:.1%}`")
        st.progress(prob_ent)
    
    with col_b:
        prob_not = result['prob_not_entailment']
        st.markdown(f"**Not Entailment:** `{prob_not:.1%}`")
        st.progress(prob_not)


def render_training_results(figures_path: str):
    """Отрисовка вкладки с результатами обучения"""
    
    st.subheader("📊 Результаты обучения модели")
    
    # Проверяем наличие графиков
    figures_exist = check_figures_exist(figures_path)
    
    if not any(figures_exist.values()):
        st.warning(f"""
        ⚠️ Графики не найдены в папке `{figures_path}`
        
        Убедитесь, что:
        1. Модель была обучена (`python scripts/train.py`)
        2. Скрипт оценки был запущен (`python scripts/evaluate.py`)
        3. Путь к графикам указан правильно
        """)
        return
    
    # Информация о модели
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ О модели:</strong><br>
        Эта модель использует Probe-подход для классификации QNLI. 
        Hidden states извлекаются из LLM (Qwen2.5-0.5B) и передаются в лёгкий 
        Transformer-классификатор для определения, содержит ли предложение ответ на вопрос.
    </div>
    """, unsafe_allow_html=True)
    
    # === Training Curves ===
    st.markdown("---")
    
    if figures_exist.get("training_curves"):
        st.markdown("""
        <div class="figure-card">
            <div class="figure-title">📈 Кривые обучения (Training Curves)</div>
            <div class="figure-description">
                Графики показывают динамику обучения модели:
                <ul>
                    <li><strong>Train/Val Loss:</strong> Как менялась функция потерь на обучающей и валидационной выборках</li>
                    <li><strong>Val Accuracy:</strong> Как росла точность на валидации</li>
                    <li><strong>Overfitting Gap:</strong> Разница между Val и Train Loss (индикатор переобучения)</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.image(
            os.path.join(figures_path, "training_curves.png"),
            use_container_width=True
        )
        
        st.markdown("""
        <div class="success-box">
            <strong>✅ Интерпретация:</strong><br>
            • Если Train и Val Loss близки — модель хорошо обобщает<br>
            • Если Val Loss растёт при падающем Train Loss — переобучение<br>
            • Красная точка на графике Accuracy — лучшая эпоха
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("📈 График training_curves.png не найден")
    
    # Confusion Matrix
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if figures_exist.get("confusion_matrix"):
            st.markdown("""
            <div class="figure-card">
                <div class="figure-title">🎯 Матрица ошибок (Confusion Matrix)</div>
                <div class="figure-description">
                    Показывает распределение предсказаний модели по классам:
                    <ul>
                        <li><strong>TP (True Positive):</strong> Верно предсказан Entailment</li>
                        <li><strong>TN (True Negative):</strong> Верно предсказан Not Entailment</li>
                        <li><strong>FP (False Positive):</strong> Ошибочно предсказан Entailment</li>
                        <li><strong>FN (False Negative):</strong> Пропущен Entailment</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.image(
                os.path.join(figures_path, "confusion_matrix.png"),
                use_container_width=True
            )
        else:
            st.warning("🎯 График confusion_matrix.png не найден")
    
    with col2:
        if figures_exist.get("confidence_analysis"):
            st.markdown("""
            <div class="figure-card">
                <div class="figure-title">📊 Анализ уверенности (Confidence Analysis)</div>
                <div class="figure-description">
                    Распределение уверенности модели для верных и ошибочных предсказаний:
                    <ul>
                        <li><strong>Зелёный:</strong> Уверенность для правильных ответов</li>
                        <li><strong>Красный:</strong> Уверенность для ошибок</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.image(
                os.path.join(figures_path, "confidence_analysis.png"),
                use_container_width=True
            )
        else:
            st.warning("📊 График confidence_analysis.png не найден")
    
    # Интерпретация
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <strong>📝 Как читать результаты:</strong><br><br>
        
        <strong>Матрица ошибок:</strong><br>
        • Идеальная модель имеет значения только на диагонали<br>
        • Нормализованная матрица показывает % от реального класса<br><br>
        
        <strong>Анализ уверенности:</strong><br>
        • Хорошо: правильные ответы с высокой уверенностью (>90%)<br>
        • Хорошо: ошибки с низкой уверенностью (~50%)<br>
        • Плохо: много ошибок с высокой уверенностью (overconfidence)
    </div>
    """, unsafe_allow_html=True)
    
    # Метрики модели
    st.markdown("---")
    st.subheader("📈 Ключевые метрики")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Пытаемся получить метрики из API
    try:
        health = check_api_health(st.session_state.get('api_url', DEFAULT_API_URL))
        if health and health.get('best_accuracy'):
            accuracy = health['best_accuracy']
            params = health['probe_parameters']
        else:
            accuracy = 0.9136 # Примерное значение
            params = 8500000
    except:
        accuracy = 0.91
        params = 8500000
    
    with col1:
        st.metric(
            label="🎯 Best Accuracy",
            value=f"{accuracy:.2%}"
        )
    
    with col2:
        st.metric(
            label="🔢 Parameters",
            value=f"{params:,}"
        )
    
    with col3:
        st.metric(
            label="📚 Dataset",
            value="QNLI"
        )
    
    with col4:
        st.metric(
            label="🤖 Base Model",
            value="Qwen2.5-0.5B"
        )


# ============================================================================
#                              ГЛАВНАЯ СТРАНИЦА
# ============================================================================

def main():
    """Главная функция приложения"""
    
    # Инициализация session state
    if 'api_url' not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL
    
    # Заголовок
    render_header()
    
    # Боковая панель
    api_url, model_loaded = render_sidebar(st.session_state.api_url)
    st.session_state.api_url = api_url
    
    # Получаем путь к графикам
    figures_path = get_figures_path()
    
    # Вкладки
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Одиночное предсказание",
        "📦 Batch обработка",
        "📚 Примеры",
        "📊 Результаты обучения"
    ])
    
    # Одиночное предсказание
    with tab1:
        # Если модель не загружена - показываем предупреждение
        if not model_loaded:
            st.warning("⚠️ API сервер недоступен или модель не загружена. Проверьте боковую панель.")
        else:
            col_input, col_result = st.columns([1, 1])
            
            with col_input:
                st.subheader("📝 Введите данные")
                
                # Загружаем примеры для выпадающего списка
                examples = get_examples(api_url)
                
                if examples:
                    example_options = ["-- Свой пример --"] + [
                        f"Пример {i+1}: {ex['expected']}"
                        for i, ex in enumerate(examples)
                    ]
                    
                    selected = st.selectbox(
                        "📋 Выберите пример или введите свой:",
                        example_options
                    )
                    
                    if selected != "-- Свой пример --":
                        idx = example_options.index(selected) - 1
                        default_q = examples[idx]["question"]
                        default_s = examples[idx]["sentence"]
                    else:
                        default_q, default_s = "", ""
                else:
                    default_q, default_s = "", ""
                
                question = st.text_area(
                    "❓ Вопрос (Question):",
                    value=default_q,
                    height=80,
                    placeholder="What is the capital of France?"
                )
                
                sentence = st.text_area(
                    "📄 Предложение (Sentence):",
                    value=default_s,
                    height=100,
                    placeholder="Paris is the capital and most populous city of France."
                )
                
                predict_btn = st.button(
                    "🚀 Анализировать",
                    type="primary",
                    use_container_width=True
                )
            
            with col_result:
                st.subheader("📊 Результат")
                
                if predict_btn:
                    if question.strip() and sentence.strip():
                        with st.spinner("⏳ Обработка..."):
                            result = predict_single(api_url, question, sentence)
                        
                        if result:
                            render_result(result)
                    else:
                        st.warning("⚠️ Заполните оба поля!")
                else:
                    st.info("👈 Введите вопрос и предложение, затем нажмите **'Анализировать'**")
    
    # Batch обработка
    with tab2:
        if not model_loaded:
            st.warning("⚠️ API сервер недоступен. Проверьте боковую панель.")
        else:
            st.subheader("📦 Batch обработка")
            
            st.info("""
            📤 **Загрузите CSV файл** с колонками:
            - `question` — вопрос
            - `sentence` — предложение
            """)
            
            uploaded_file = st.file_uploader(
                "Выберите CSV файл:",
                type=['csv'],
                help="Файл должен содержать колонки 'question' и 'sentence'"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Проверка колонок
                    if 'question' not in df.columns or 'sentence' not in df.columns:
                        st.error("❌ CSV должен содержать колонки 'question' и 'sentence'")
                        return
                    
                    st.success(f"✅ Загружено {len(df)} примеров")
                    
                    # Предпросмотр
                    with st.expander("👀 Предпросмотр данных"):
                        st.dataframe(df.head(10))
                    
                    # Кнопка обработки
                    if st.button("🚀 Обработать все", type="primary"):
                        items = df[['question', 'sentence']].to_dict('records')
                        
                        # Прогресс
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Обработка батчами по 10
                        batch_size = 10
                        all_results = []
                        
                        for i in range(0, len(items), batch_size):
                            batch = items[i:i + batch_size]
                            current_end = min(i + batch_size, len(items))
                            
                            status_text.text(f"⏳ Обработка {i+1}-{current_end} из {len(items)}...")
                            
                            result = predict_batch(api_url, batch)
                            
                            if result:
                                all_results.extend(result['results'])
                            else:
                                st.error(f"Ошибка при обработке батча {i+1}-{current_end}")
                                break
                            
                            progress_bar.progress(current_end / len(items))
                        
                        status_text.empty()
                        progress_bar.empty()
                        
                        # Результаты
                        if len(all_results) == len(df):
                            st.success(f"✅ Обработано {len(all_results)} примеров!")
                            
                            # Добавляем результаты в DataFrame
                            df['prediction'] = [r['label'] for r in all_results]
                            df['confidence'] = [r['confidence'] for r in all_results]
                            df['prob_entailment'] = [r['prob_entailment'] for r in all_results]
                            df['prob_not_entailment'] = [r['prob_not_entailment'] for r in all_results]
                            
                            # Статистика
                            st.markdown("### 📊 Статистика")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            entail_count = (df['prediction'] == 'entailment').sum()
                            not_entail_count = (df['prediction'] == 'not_entailment').sum()
                            avg_conf = df['confidence'].mean()
                            
                            col1.metric("Всего", len(df))
                            col2.metric("Entailment", entail_count)
                            col3.metric("Not Entailment", not_entail_count)
                            col4.metric("Avg Confidence", f"{avg_conf:.1%}")
                            
                            # Таблица результатов
                            st.markdown("### 📋 Результаты")
                            st.dataframe(df, use_container_width=True)
                            
                            # Кнопка скачивания
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Скачать результаты (CSV)",
                                data=csv,
                                file_name="qnli_results.csv",
                                mime="text/csv"
                            )
                        
                except Exception as e:
                    st.error(f"❌ Ошибка чтения файла: {e}")
    
    # Примеры
    with tab3:
        if not model_loaded:
            st.warning("⚠️ API сервер недоступен. Проверьте боковую панель.")
        else:
            st.subheader("📚 Примеры для тестирования")
            
            examples = get_examples(api_url)
            
            if examples:
                for i, ex in enumerate(examples):
                    emoji = "✅" if ex['expected'] == "entailment" else "❌"
                    
                    with st.expander(f"{emoji} Пример {i+1}: {ex['expected']}"):
                        st.markdown(f"**❓ Вопрос:**")
                        st.info(ex['question'])
                        
                        st.markdown(f"**📄 Предложение:**")
                        st.info(ex['sentence'])
                        
                        st.markdown(f"**🎯 Ожидаемый результат:** `{ex['expected']}`")
                        
                        # Кнопка проверки
                        if st.button(f"🧪 Проверить", key=f"example_{i}"):
                            with st.spinner("Проверка..."):
                                result = predict_single(api_url, ex['question'], ex['sentence'])
                            
                            if result:
                                is_correct = result['label'] == ex['expected']
                                
                                if is_correct:
                                    st.success(
                                        f"✅ **Верно!** "
                                        f"Предсказание: `{result['label']}` "
                                        f"(уверенность: {result['confidence']:.1%})"
                                    )
                                else:
                                    st.error(
                                        f"❌ **Ошибка!** "
                                        f"Предсказание: `{result['label']}`, "
                                        f"Ожидалось: `{ex['expected']}` "
                                        f"(уверенность: {result['confidence']:.1%})"
                                    )
            else:
                st.warning("Не удалось загрузить примеры из API")
    
    # Результаты обучения
    with tab4:
        render_training_results(figures_path)
    
    st.markdown("---")
    st.markdown(
        """
        <p style="text-align: center; color: #888; font-size: 0.9rem;">
            QNLI Probe Demo | FastAPI + Streamlit | 
            Powered by Qwen2.5 + Custom Transformer Probe
        </p>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
#                              ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    main()