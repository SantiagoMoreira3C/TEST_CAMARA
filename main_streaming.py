from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
from detector import YOLODetector

app = FastAPI()

# Cargar modelo YOLO
detector = YOLODetector(model_name="yolov8n.pt", confidence_threshold=0.5)

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Redimensionar y detectar
        small_frame = cv2.resize(frame, (640, 480))
        detections = detector.detect(small_frame)
        annotated_frame = detector.draw_detections(small_frame, detections)
        
        # Codificar el frame a JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        # Generar stream MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    # Escucha en todas las interfaces de red en el puerto 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
