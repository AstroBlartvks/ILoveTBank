# Используем официальный Python образ
FROM python:3.13-slim

# Устанавливаем системные зависимости для OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*


# Создаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY main.py .
COPY validate.py .

COPY main.py /app/main.py
COPY validate.py /app/validate.py

# Создаем директорию для модели
RUN mkdir -p /app/models

# Копируем модель (можно заменить на загрузку извне)
COPY model.pt /app/models/model.pt

# Добавляем переменную среды для модели
ENV MODEL_PATH=/app/models/model.pt

# Создаем непривилегированного пользователя
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Открываем порт 8000
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]