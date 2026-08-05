import logging
import base64
import cv2
import numpy as np
from io import BytesIO
from typing import Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


class CameraFrameHandler:
    """
    Manejador de frames de cámara.
    Convierte formatos entre base64, binary y arrays de OpenCV.
    """

    @staticmethod
    def base64_to_frame(base64_string: str) -> Optional[np.ndarray]:
        """
        Convierte string base64 a frame de OpenCV.

        Args:
            base64_string: String en base64 (puede incluir metadata JPEG)

        Returns:
            np.ndarray o None si falla
        """
        try:
            # Eliminar prefijo de data URI si existe (data:image/jpeg;base64,)
            if "," in base64_string:
                base64_string = base64_string.split(",", 1)[1]

            # Decodificar base64
            image_data = base64.b64decode(base64_string)

            # Convertir a imagen
            image = Image.open(BytesIO(image_data)).convert("RGB")

            # Convertir PIL a OpenCV (BGR)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            return frame

        except Exception as e:
            logger.error(f"Error convirtiendo base64 a frame: {e}")
            return None

    @staticmethod
    def frame_to_base64(frame: np.ndarray, quality: int = 80) -> Optional[str]:
        """
        Convierte frame de OpenCV a string base64.

        Args:
            frame: Array de imagen OpenCV (BGR)
            quality: Calidad de compresión JPEG (0-100)

        Returns:
            String base64 o None si falla
        """
        try:
            # Convertir BGR a RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Comprimir a JPEG
            _, buffer = cv2.imencode(".jpg", rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

            # Convertir a base64
            base64_string = base64.b64encode(buffer).decode("utf-8")

            return f"data:image/jpeg;base64,{base64_string}"

        except Exception as e:
            logger.error(f"Error convirtiendo frame a base64: {e}")
            return None

    @staticmethod
    def resize_frame(frame: np.ndarray, target_width: int = 416, maintain_aspect: bool = True) -> np.ndarray:
        """
        Redimensiona frame manteniendo aspecto.

        Args:
            frame: Frame original
            target_width: Ancho objetivo (altura se ajusta automáticamente)
            maintain_aspect: Si es True, mantiene aspecto; si False, fuerza tamaño

        Returns:
            Frame redimensionado
        """
        try:
            if frame is None or frame.size == 0:
                return frame

            height, width = frame.shape[:2]

            if maintain_aspect:
                # Mantener aspecto
                aspect_ratio = height / width
                new_height = int(target_width * aspect_ratio)
                new_size = (target_width, new_height)
            else:
                # Forzar tamaño cuadrado (mejor para YOLO)
                new_size = (target_width, target_width)

            resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
            return resized

        except Exception as e:
            logger.error(f"Error redimensionando frame: {e}")
            return frame

    @staticmethod
    def normalize_frame(frame: np.ndarray) -> np.ndarray:
        """
        Normaliza valores de píxeles a 0-1 (útil para modelos).

        Args:
            frame: Frame original (0-255)

        Returns:
            Frame normalizado (0-1)
        """
        return frame.astype(np.float32) / 255.0

    @staticmethod
    def denormalize_frame(frame: np.ndarray) -> np.ndarray:
        """
        Desnormaliza frame (0-1) a (0-255).

        Args:
            frame: Frame normalizado

        Returns:
            Frame 0-255
        """
        return (frame * 255).astype(np.uint8)

    @staticmethod
    def apply_brightness_contrast(
        frame: np.ndarray, brightness: int = 0, contrast: float = 1.0
    ) -> np.ndarray:
        """
        Aplica ajustes de brillo y contraste.

        Args:
            frame: Frame original
            brightness: Ajuste de brillo (-100 a 100)
            contrast: Multiplicador de contraste (0.5 a 2.0)

        Returns:
            Frame ajustado
        """
        try:
            adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            return adjusted
        except Exception as e:
            logger.error(f"Error ajustando brillo/contraste: {e}")
            return frame

    @staticmethod
    def get_frame_info(frame: np.ndarray) -> dict:
        """
        Obtiene información sobre el frame.

        Args:
            frame: Frame a analizar

        Returns:
            Dict con información
        """
        if frame is None or frame.size == 0:
            return {"width": 0, "height": 0, "channels": 0, "dtype": "unknown"}

        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) == 3 else 1

        return {
            "width": width,
            "height": height,
            "channels": channels,
            "dtype": str(frame.dtype),
            "size_mb": round(frame.nbytes / (1024 * 1024), 2),
        }
