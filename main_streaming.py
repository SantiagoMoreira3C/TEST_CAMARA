"""
🎥 YOLO Person Detector - Webcam en Vivo (Sin Web)
Detecta personas en tiempo real desde tu cámara web
"""

import cv2
from detector import YOLODetector

def main():
    print("\n" + "="*60)
    print(" 🎥 YOLO Person Detector - Webcam en Vivo")
    print("="*60)
    print("\nPresiona 'q' para salir\n")
    
    # Inicializar detector
    print("⏳ Cargando modelo YOLOv8...")
    detector = YOLODetector(model_name="yolov8n.pt", confidence_threshold=0.5)
    print("✅ Modelo cargado!\n")
    
    # Abrir cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        return
    
    print("📷 Cámara activada. Presiona 'q' para salir...\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error leyendo frame")
            break
        
        frame_count += 1
        
        # Redimensionar para procesamiento más rápido
        small_frame = cv2.resize(frame, (640, 480))
        
        # Detectar personas
        detections = detector.detect(small_frame)
        
        # Dibujar resultados
        annotated_frame = detector.draw_detections(small_frame, detections)
        
        # Mostrar información
        person_count = detections['person_count']
        avg_conf = detections['avg_confidence']
        
        # Texto con info
        info_text = f"Personas: {person_count} | Confianza: {avg_conf:.2f} | Frame: {frame_count}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar en ventana
        cv2.imshow("🎥 YOLO Person Detector", annotated_frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n✅ Saliendo...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Listo!")

if __name__ == "__main__":
    main()
