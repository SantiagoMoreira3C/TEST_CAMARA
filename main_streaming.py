"""
🎥 YOLO Detector - Cliente envía cámara al servidor
El teléfono/PC envía su cámara → Servidor procesa → Devuelve resultado
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from detector import YOLODetector
import cv2
import numpy as np
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLO Detector - Client Camera")

# CORS para acceso desde cualquier dispositivo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = None

@app.on_event("startup")
async def startup():
    global detector
    logger.info("⏳ Inicializando YOLO...")
    detector = YOLODetector(model_name="yolov8n.pt", confidence_threshold=0.5)
    logger.info("✅ YOLO listo!")

@app.get("/")
async def index():
    """Página con acceso a cámara del cliente"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎥 YOLO Person Detector</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                max-width: 100%;
                width: 800px;
            }
            h1 {
                color: #667eea;
                margin-bottom: 20px;
                text-align: center;
            }
            .video-section {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 20px 0;
            }
            @media (max-width: 768px) {
                .video-section {
                    grid-template-columns: 1fr;
                }
            }
            video, canvas {
                width: 100%;
                border-radius: 10px;
                border: 3px solid #667eea;
                background: black;
            }
            .controls {
                display: flex;
                gap: 10px;
                margin: 20px 0;
            }
            button {
                flex: 1;
                padding: 12px 20px;
                font-size: 16px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
                transition: 0.3s;
            }
            .btn-start {
                background: #4CAF50;
                color: white;
            }
            .btn-start:hover {
                background: #45a049;
            }
            .btn-stop {
                background: #f44336;
                color: white;
            }
            .btn-stop:hover {
                background: #da190b;
            }
            .status {
                background: #e8f5e9;
                color: #2e7d32;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
                font-weight: bold;
                text-align: center;
            }
            .info {
                background: #f3e5f5;
                color: #6a1b9a;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
                font-size: 14px;
            }
            .stats {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin: 15px 0;
            }
            .stat-box {
                background: #f0f0f0;
                padding: 10px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-box strong {
                color: #667eea;
                display: block;
                font-size: 18px;
            }
            .stat-box span {
                color: #666;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 YOLO Person Detector</h1>
            
            <div class="status" id="status">
                📱 Preparado para detectar personas
            </div>
            
            <div class="video-section">
                <div>
                    <h3 style="text-align: center; color: #667eea; margin-bottom: 10px;">📹 Tu Cámara</h3>
                    <video id="video" autoplay playsinline></video>
                </div>
                <div>
                    <h3 style="text-align: center; color: #667eea; margin-bottom: 10px;">🎯 Detecciones</h3>
                    <canvas id="canvas"></canvas>
                </div>
            </div>
            
            <div class="controls">
                <button class="btn-start" onclick="startDetection()">▶️ Iniciar Detección</button>
                <button class="btn-stop" onclick="stopDetection()">⏹️ Detener</button>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <strong id="personCount">0</strong>
                    <span>Personas detectadas</span>
                </div>
                <div class="stat-box">
                    <strong id="confidence">0%</strong>
                    <span>Confianza promedio</span>
                </div>
            </div>
            
            <div class="info">
                <strong>ℹ️ Cómo funciona:</strong><br>
                1. Haz clic en "Iniciar Detección"<br>
                2. Permite acceso a tu cámara<br>
                3. Los frames se envían al servidor<br>
                4. Se procesan con YOLO<br>
                5. Ves el resultado en tiempo real<br>
                <br>
                <strong>🎨 Colores:</strong><br>
                🟢 Verde = Alta confianza (≥75%) | 🟠 Naranja = Media (50-75%) | 🔴 Rojo = Baja (<50%)
            </div>
        </div>

        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const status = document.getElementById('status');
            const personCountEl = document.getElementById('personCount');
            const confidenceEl = document.getElementById('confidence');
            
            let isRunning = false;
            let stream = null;

            async function startDetection() {
                try {
                    status.textContent = '📹 Solicitando acceso a cámara...';
                    
                    // Obtener cámara del cliente
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: { width: 640, height: 480 }
                    });
                    
                    video.srcObject = stream;
                    canvas.width = 640;
                    canvas.height = 480;
                    
                    isRunning = true;
                    status.textContent = '✅ Cámara conectada | 📊 Procesando...';
                    
                    processFrames();
                } catch (err) {
                    status.textContent = '❌ Error: ' + err.message;
                    status.style.background = '#ffebee';
                    status.style.color = '#c62828';
                }
            }

            async function processFrames() {
                if (!isRunning) return;
                
                try {
                    // Dibujar frame en canvas
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    // Obtener datos de imagen
                    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
                    
                    // Enviar al servidor
                    const formData = new FormData();
                    formData.append('file', blob, 'frame.jpg');
                    
                    const response = await fetch('/detect', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) throw new Error('Error en servidor');
                    
                    const result = await response.json();
                    
                    // Mostrar resultado procesado
                    const img = new Image();
                    img.onload = () => {
                        ctx.drawImage(img, 0, 0);
                    };
                    img.src = result.frame;
                    
                    // Actualizar estadísticas
                    personCountEl.textContent = result.person_count;
                    confidenceEl.textContent = (result.avg_confidence * 100).toFixed(1) + '%';
                    
                    // Continuar procesando
                    setTimeout(processFrames, 100);
                } catch (err) {
                    console.error('Error procesando frame:', err);
                    setTimeout(processFrames, 100);
                }
            }

            function stopDetection() {
                isRunning = false;
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    video.srcObject = null;
                }
                status.textContent = '⏹️ Detención completada';
                status.style.background = '#fff3e0';
                status.style.color = '#e65100';
                personCountEl.textContent = '0';
                confidenceEl.textContent = '0%';
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }

            // Iniciar automáticamente
            window.addEventListener('load', () => {
                status.textContent = '✅ Página cargada | Haz clic en "Iniciar Detección"';
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/detect")
async def detect(file):
    """Recibe frame, detecta personas, devuelve resultado"""
    try:
        # Leer archivo
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JSONResponse({"error": "Frame inválido"}, status_code=400)
        
        # Redimensionar
        frame = cv2.resize(frame, (640, 480))
        
        # Detectar
        detections = detector.detect(frame)
        
        # Dibujar
        annotated_frame = detector.draw_detections(frame, detections)
        
        # Codificar a base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_base64 = base64.b64encode(buffer).decode()
        
        return {
            "person_count": detections['person_count'],
            "avg_confidence": detections['avg_confidence'],
            "frame": f"data:image/jpeg;base64,{frame_base64}"
        }
    except Exception as e:
        logger.error(f"Error en detección: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print(" 🎥 YOLO Detector - Cliente Envía Cámara")
    print("="*60)
    print("\n📱 Accede desde:")
    print("   • Teléfono: http://TU_IP:3030")
    print("   • PC: http://localhost:3030")
    print("\n✨ La cámara del CLIENTE se procesa en el SERVIDOR")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=3030)
