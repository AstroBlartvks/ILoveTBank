from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO
from typing import List, Optional
from pydantic import BaseModel, Field
import time
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="T-Bank Logo Detection API",
    description="REST API для детекции логотипа Т-Банка на изображениях",
    version="1.0.0"
)

# Получаем путь к модели из переменной окружения или используем дефолтный
model_path = os.getenv("MODEL_PATH", "model.pt")
logger.info(f"Loading model from: {model_path}")
model = YOLO(model_path)

class BoundingBox(BaseModel):
    """Абсолютные координаты BoundingBox"""
    x_min: int = Field(..., description="Левая координата", ge=0)
    y_min: int = Field(..., description="Верхняя координата", ge=0)
    x_max: int = Field(..., description="Правая координата", ge=0)
    y_max: int = Field(..., description="Нижняя координата", ge=0)

class Detection(BaseModel):
    """Результат детекции одного логотипа"""
    bbox: BoundingBox = Field(..., description="Результат детекции")

class DetectionResponse(BaseModel):
    """Ответ API с результатами детекции"""
    detections: List[Detection] = Field(..., description="Список найденных логотипов")

class ErrorResponse(BaseModel):
    """Ответ при ошибке"""
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Дополнительная информация")

def draw_boxes(image: np.ndarray, boxes: List[BoundingBox]):
    for box in boxes:
        cv2.rectangle(image, (box.x_min, box.y_min), (box.x_max, box.y_max), (0,255,0), 2)
    return image

@app.post("/detect", response_model=DetectionResponse)
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении

    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)

    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """

    if file.content_type not in ["image/jpeg", "image/png", "image/bmp", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Supported formats: JPEG, PNG, BMP, WEBP"
        )

    start_time = time.time()
    logger.info(f"Processing image: {file.filename}, size: {file.size} bytes")

    try:
        data = await file.read()
        img_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file or corrupted data"
            )

        # Model prediction with timing
        prediction_start = time.time()
        results = model.predict(img, conf=0.3)
        prediction_time = time.time() - prediction_start

        detections = []
        if len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                    if int(cls) == 0:  
                        bbox = BoundingBox(
                            x_min=max(0, int(box[0])),
                            y_min=max(0, int(box[1])),
                            x_max=max(0, int(box[2])),
                            y_max=max(0, int(box[3]))
                        )
                        detections.append(Detection(bbox=bbox))

        processing_time = time.time() - start_time
        logger.info(f"Image processed in {processing_time:.3f}s (prediction: {prediction_time:.3f}s), found {len(detections)} logos")

        return DetectionResponse(detections=detections)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    data = await file.read()
    img_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")

    results = model.predict(img, conf=0.3)
    boxes = []
    result = results[0]
    for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        if int(cls) == 0:
            boxes.append([int(x) for x in box[:4]])
    img_with_boxes = draw_boxes(img.copy(), [BoundingBox(x_min=b[0], y_min=b[1], x_max=b[2], y_max=b[3]) for b in boxes])

    _, encoded = cv2.imencode(".png", img_with_boxes)
    return StreamingResponse(BytesIO(encoded.tobytes()), media_type="image/png")
