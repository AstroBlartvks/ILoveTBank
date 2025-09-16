#!/usr/bin/env python3
"""
Исправленный скрипт для валидации модели детекции логотипа Т-Банка
Работает с YOLO форматом аннотаций (.txt файлы)
Рассчитывает метрики Precision, Recall, F1-score при IoU=0.5
"""

import os
import json
import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Вычисляет IoU (Intersection over Union) между двумя bounding boxes

    Args:
        box1: [x_min, y_min, x_max, y_max]
        box2: [x_min, y_min, x_max, y_max]

    Returns:
        IoU значение от 0 до 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Находим координаты пересечения
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    if xi_max <= xi_min or yi_max <= yi_min:
        return 0.0

    # Площадь пересечения
    intersection_area = (xi_max - xi_min) * (yi_max - yi_min)

    # Площади исходных box'ов
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Площадь объединения
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0

def yolo_to_absolute(yolo_bbox: List[float], img_width: int, img_height: int) -> List[int]:
    """
    Конвертирует YOLO формат (normalized) в абсолютные координаты
    YOLO: [center_x, center_y, width, height] (normalized 0-1)
    Absolute: [x_min, y_min, x_max, y_max]
    """
    center_x, center_y, width, height = yolo_bbox

    # Денормализация
    center_x *= img_width
    center_y *= img_height
    width *= img_width
    height *= img_height

    # Конвертация в x_min, y_min, x_max, y_max
    x_min = int(center_x - width / 2)
    y_min = int(center_y - height / 2)
    x_max = int(center_x + width / 2)
    y_max = int(center_y + height / 2)

    return [x_min, y_min, x_max, y_max]

def evaluate_model(model_path: str, validation_dir: str, confidence_threshold: float = 0.3, iou_threshold: float = 0.5) -> Dict:
    """
    Оценивает модель на валидационном наборе данных

    Args:
        model_path: Путь к модели YOLO
        validation_dir: Директория с валидационными данными
        confidence_threshold: Порог уверенности для детекции
        iou_threshold: Порог IoU для считания детекции правильной

    Returns:
        Словарь с метриками
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Поиск изображений и аннотаций
    images_dir = Path(validation_dir) / "images"
    annotations_dir = Path(validation_dir) / "labels"  # YOLO использует labels вместо annotations

    # Проверяем альтернативную структуру
    if not annotations_dir.exists():
        annotations_dir = Path(validation_dir) / "annotations"

    if not images_dir.exists():
        raise FileNotFoundError(f"Directory {images_dir} not found")

    print(f"Images directory: {images_dir}")
    print(f"Annotations directory: {annotations_dir}")

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
    print(f"Found {len(image_files)} images for validation")

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    images_processed = 0
    images_with_annotations = 0
    total_ground_truth_boxes = 0

    results_for_visualization = []

    for image_file in image_files:
        # Загрузка изображения
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Warning: Could not load image {image_file}")
            continue

        # Предсказание модели
        results = model.predict(image, conf=confidence_threshold, verbose=False)
        predicted_boxes = []

        if len(results) > 0 and results[0].boxes is not None:
            for box, cls in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.cls.cpu().numpy()):
                if int(cls) == 0:  # Класс логотипа Т-Банка
                    predicted_boxes.append([int(x) for x in box])

        # Загрузка ground truth аннотаций в YOLO формате
        annotation_file = annotations_dir / f"{image_file.stem}.txt"
        ground_truth_boxes = []

        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            images_with_annotations += 1
            with open(annotation_file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if class_id == 0:  # Только класс логотипа Т-Банка
                                yolo_bbox = [float(x) for x in parts[1:5]]
                                absolute_bbox = yolo_to_absolute(yolo_bbox, image.shape[1], image.shape[0])
                                ground_truth_boxes.append(absolute_bbox)

        total_ground_truth_boxes += len(ground_truth_boxes)

        # Сопоставление предсказанных и истинных боксов
        matched_gt = set()
        matched_pred = set()

        for i, pred_box in enumerate(predicted_boxes):
            best_iou = 0
            best_gt_idx = -1

            for j, gt_box in enumerate(ground_truth_boxes):
                if j in matched_gt:
                    continue

                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold:
                true_positives += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
            else:
                false_positives += 1

        # Подсчет пропущенных ground truth боксов
        false_negatives += len(ground_truth_boxes) - len(matched_gt)

        # Сохранение результатов для визуализации
        if len(ground_truth_boxes) > 0 or len(predicted_boxes) > 0:  # Только интересные случаи
            results_for_visualization.append({
                'image_path': str(image_file),
                'predicted_boxes': predicted_boxes,
                'ground_truth_boxes': ground_truth_boxes,
                'image': image
            })

        images_processed += 1
        if images_processed % 50 == 0:
            print(f"Processed {images_processed}/{len(image_files)} images...")

    print(f"\nValidation Summary:")
    print(f"Images processed: {images_processed}")
    print(f"Images with annotations: {images_with_annotations}")
    print(f"Total ground truth boxes: {total_ground_truth_boxes}")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")

    # Вычисление метрик
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_images': len(image_files),
        'images_with_annotations': images_with_annotations,
        'total_ground_truth_boxes': total_ground_truth_boxes,
        'results_for_visualization': results_for_visualization
    }

    return metrics

def visualize_results(metrics: Dict, output_dir: str = "validation_results", max_images: int = 10):
    """
    Создает визуализацию результатов детекции

    Args:
        metrics: Результаты валидации
        output_dir: Директория для сохранения результатов
        max_images: Максимальное количество изображений для визуализации
    """
    os.makedirs(output_dir, exist_ok=True)

    results = metrics['results_for_visualization'][:max_images]
    print(f"Creating visualizations for {len(results)} images...")

    for i, result in enumerate(results):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Отображение изображения
        image_rgb = cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)

        # Отрисовка ground truth боксов (зеленые)
        for bbox in result['ground_truth_boxes']:
            x_min, y_min, x_max, y_max = bbox
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   linewidth=2, edgecolor='green', facecolor='none', label='Ground Truth')
            ax.add_patch(rect)

        # Отрисовка предсказанных боксов (красные)
        for bbox in result['predicted_boxes']:
            x_min, y_min, x_max, y_max = bbox
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   linewidth=2, edgecolor='red', facecolor='none', label='Predicted')
            ax.add_patch(rect)

        ax.set_title(f"Image {i+1}: {Path(result['image_path']).name}\\nGT: {len(result['ground_truth_boxes'])}, Pred: {len(result['predicted_boxes'])}")
        ax.axis('off')

        # Легенда (избегаем дублирования)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"validation_result_{i+1}.png"), dpi=150, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Validate T-Bank logo detection model")
    parser.add_argument("--model", default="model.pt", help="Path to YOLO model")
    parser.add_argument("--data", default="dataset/valid", help="Path to validation dataset directory")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument("--output", default="validation_results", help="Output directory for results")

    args = parser.parse_args()

    print("🔍 Starting model validation...")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print("-" * 50)

    try:
        # Валидация модели
        metrics = evaluate_model(args.model, args.data, args.conf, args.iou)

        # Вывод результатов
        print("📊 Validation Results:")
        print(f"Total images processed: {metrics['total_images']}")
        print(f"Images with annotations: {metrics['images_with_annotations']}")
        print(f"Total ground truth boxes: {metrics['total_ground_truth_boxes']}")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")

        # Сохранение метрик в JSON
        output_path = os.path.join(args.output, "metrics.json")
        os.makedirs(args.output, exist_ok=True)

        metrics_to_save = {k: v for k, v in metrics.items() if k != 'results_for_visualization'}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)

        print(f"📁 Metrics saved to: {output_path}")

        # Создание визуализации
        if metrics['results_for_visualization']:
            print("🖼️  Creating visualizations...")
            visualize_results(metrics, args.output)
            print(f"📁 Visualizations saved to: {args.output}")
        else:
            print("⚠️  No results for visualization (no images with detections or ground truth)")

    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("✅ Validation completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())