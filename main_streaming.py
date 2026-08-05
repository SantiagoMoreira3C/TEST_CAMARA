"""
🎥 YOLO Person Detector - Streaming en Navegador
Accede desde: http://localhost:3030 o http://TU_IP:3030
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import asyncio
from detector import YOLODetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLO Detector - Streaming")

# CORS para acceso desde cualquier IP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
detector = None
cap = None

@app.on_event("startup")
async def startup():
    global detector, cap
    logger.info("Inicializando YOLO...")
    detector = YOLODetector(model_name="yolov8n.pt", confidence_threshold=0.5)
    logger.info("✅ YOLO listo!")
    
    logger.info("Abriendo cámara...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("❌ No se pudo abrir la cámara")
    else:
        logger.info("✅ Cámara lista!")

@app.on_event("shutdown")
async def shutdown():
    global cap
    if cap:
        cap.release()

def generate_frames():
    """Genera frames del video con detecciones"""
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Error leyendo frame")
            break
        
        frame_count += 1
        
        # Redimensionar
        frame = cv2.resize(frame, (640, 480))
        
        # Detectar
        detections = detector.detect(frame)
        
        # Dibujar
        annotated_frame = detector.draw_detections(frame, detections)
        
        # Agregar información
        person_count = detections['person_count']
        avg_conf = detections['avg_confidence']
        info_text = f"Personas: {person_count} | Confianza: {avg_conf:.2f} | Frame: {frame_count}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Codificar a JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        # Formato MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' 
               + frame_bytes + b'\r\n')

@app.get("/")
async def index():
    """Página principal con video en vivo"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎥 YOLO Person Detector</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
            }
            .container {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                text-align: center;
                max-width: 800px;
            }
            h1 {
                color: #667eea;
                margin: 0 0 20px 0;
            }
            video, img {
                max-width: 100%;
                border-radius: 10px;
                border: 3px solid #667eea;
                margin: 20px 0;
            }
            .status {
                background: #e8f5e9;
                color: #2e7d32;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
                font-weight: bold;
            }
            .info {
                background: #f3e5f5;
                color: #6a1b9a;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
                text-align: left;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 YOLO Person Detector - En Vivo</h1>
            
            <div class="status">
                ✅ Servidor activo | 📷 Cámara conectada
            </div>
            
            <img src="/video" alt="Video en vivo">
            
            <div class="info">
                <strong>ℹ️ Información:</strong><br>
                • Detección de personas en tiempo real<br>
                • Verde = Alta confianza (≥75%)<br>
                • Naranja = Confianza media (50-75%)<br>
                • Rojo = Baja confianza (<50%)<br>
                <br>
                <strong>🌐 Acceso desde red local:</strong><br>
                http://TU_IP:3030
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/video")
async def video_feed():
    """Stream de video MJPEG"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print(" 🎥 YOLO Detector - Streaming en Navegador")
    print("="*60)
    print("\n📱 Accede desde:")
    print("   • Local: http://localhost:3030")
    print("   • Red:   http://TU_IP:3030")
    print("\nPara obtener tu IP, ejecuta: ipconfig")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=3030)
