# T-Bank Logo Detection API - Docker запуск

## Быстрый запуск одной командой

```bash
docker-compose up --build -d
```

## Проверка работы API

API будет доступен по адресу: http://localhost:8000

- Документация Swagger UI: http://localhost:8000/docs
- Эндпоинты:
  - POST `/detect` - детекция логотипа (возвращает JSON)
  - POST `/detect/image` - детекция с возвратом изображения

## Остановка контейнера

```bash
docker-compose down
```

## Альтернативный запуск через Docker

```bash
# Сборка образа
docker build -t tbank-logo-api .

# Запуск контейнера
docker run -d -p 8000:8000 --name tbank-logo-detection tbank-logo-api
```

## Тестирование API

```bash
# Тест с изображением
curl -X POST "http://localhost:8000/detect" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```