import logging
from typing import Dict, List, Tuple
from collections import deque
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    Detector de personas usando YOLOv8.
    Detecta y cuenta personas en frames de video.
    """

    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Inicializa el detector YOLO.

        Args:
            model_name: Nombre del modelo YOLO a usar (default: yolov8n.pt - Nano)
            confidence_threshold: Umbral de confianza mínimo para detectar (0-1)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.person_class_id = 0  # En YOLO, la clase 0 es "persona"
        self.detection_history = deque(maxlen=50)  # Últimas 50 detecciones

        self._load_model()

    def _load_model(self):
        """Carga el modelo YOLO."""
        try:
            logger.info(f"Cargando modelo {self.model_name}...")
            self.model = YOLO(self.model_name)
            logger.info("Modelo YOLO cargado exitosamente")
        except Exception as e:
            logger.error(f"Error al cargar modelo YOLO: {e}")
            raise

    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detecta personas en un frame.

        Args:
            frame: Array de imagen (OpenCV format - BGR)

        Returns:
            Dict con:
                - person_count: int (número de personas detectadas)
                - detections: List[Dict] (detalles de cada detección)
                - confidence_scores: List[float] (confianza de cada detección)
                - avg_confidence: float (confianza promedio)
                - timestamp: str (timestamp de la detección)
                - frame_shape: Tuple (alto, ancho del frame)
        """
        try:
            if frame is None or frame.size == 0:
                return self._empty_result()

            timestamp = datetime.now().isoformat()

            # Ejecutar detección
            results = self.model(frame, verbose=False, conf=self.confidence_threshold)

            person_detections = []
            confidence_scores = []

            # Procesar resultados
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls[0])

                        # Solo nos interesan las personas (clase 0)
                        if class_id == self.person_class_id:
                            confidence = float(box.conf[0])
                            confidence_scores.append(confidence)

                            # Coordenadas del bounding box
                            bbox = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, bbox)

                            detection_info = {
                                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                "confidence": round(confidence, 3),
                                "center": {
                                    "x": int((x1 + x2) / 2),
                                    "y": int((y1 + y2) / 2),
                                },
                            }
                            person_detections.append(detection_info)

            person_count = len(person_detections)
            avg_confidence = (
                round(sum(confidence_scores) / len(confidence_scores), 3)
                if confidence_scores
                else 0.0
            )

            result_dict = {
                "person_count": person_count,
                "detections": person_detections,
                "confidence_scores": [round(c, 3) for c in confidence_scores],
                "avg_confidence": avg_confidence,
                "timestamp": timestamp,
                "frame_shape": frame.shape[:2],  # (altura, ancho)
            }

            # Guardar en historial
            self.detection_history.append(result_dict)

            return result_dict

        except Exception as e:
            logger.error(f"Error en detección: {e}")
            return self._empty_result()

    def _empty_result(self) -> Dict:
        """Retorna un resultado vacío."""
        return {
            "person_count": 0,
            "detections": [],
            "confidence_scores": [],
            "avg_confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
            "frame_shape": (0, 0),
        }

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas del historial de detecciones.

        Returns:
            Dict con estadísticas agregadas
        """
        if not self.detection_history:
            return {
                "total_frames": 0,
                "avg_person_count": 0,
                "max_person_count": 0,
                "min_person_count": 0,
                "avg_confidence": 0,
            }

        counts = [d["person_count"] for d in self.detection_history]
        confidences = [d["avg_confidence"] for d in self.detection_history if d["avg_confidence"] > 0]

        return {
            "total_frames": len(self.detection_history),
            "avg_person_count": round(sum(counts) / len(counts), 2),
            "max_person_count": max(counts),
            "min_person_count": min(counts),
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0,
        }

    def draw_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """
        Dibuja bounding boxes sobre el frame.

        Args:
            frame: Array de imagen original
            detections: Resultado de detect()

        Returns:
            Frame con bounding boxes dibujados
        """
        annotated_frame = frame.copy()

        for detection in detections["detections"]:
            bbox = detection["bbox"]
            confidence = detection["confidence"]

            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

            # Color: verde para buena confianza, naranja para media, rojo para baja
            if confidence >= 0.75:
                color = (0, 255, 0)  # Verde
            elif confidence >= 0.5:
                color = (0, 165, 255)  # Naranja
            else:
                color = (0, 0, 255)  # Rojo

            # Dibujar bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Dibujar etiqueta con confianza
            label = f"Person {confidence:.2f}"
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Dibujar contador total
        total_count = detections["person_count"]
        avg_conf = detections["avg_confidence"]
        counter_text = f"Total: {total_count} | Avg Conf: {avg_conf:.3f}"
        cv2.putText(
            annotated_frame,
            counter_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        return annotated_frame
