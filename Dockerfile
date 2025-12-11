FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Используем ТАКУЮ ЖЕ версию PyTorch как у тебя локально (но CPU)
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Остальные зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY src/ src/
COPY demo/ demo/

RUN mkdir -p outputs results/best_model/figures

EXPOSE 8000 8501

CMD ["uvicorn", "demo.api.main:app", "--host", "0.0.0.0", "--port", "8000"]