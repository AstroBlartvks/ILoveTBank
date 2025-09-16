# T-Bank Logo Detection API - Docker запуск

## Быстрый запуск одной командой из Docker Hub 

```bash
# Придется подождать. пока скачает образ (8 Гб)
docker run -p 8000:8000 astroblartvks/model-tbank-logo-api:latest
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
# Сборка образа, НО СНАЧАЛА СКАЧАТЬ model.pt, см. README.md
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

