"""
app.py — API FastAPI para tudosis-scripts
=========================================
Envuelve newsletter_video_gen.py y carousel_gen.py como endpoints HTTP.
n8n llama a esta API en lugar de usar Execute Command.

Endpoints:
  GET  /health               → Estado del servicio
  POST /generate-video       → Genera MP4 (WF05)
  POST /generate-carousel    → Genera 5 JPG como JSON con base64 (WF06)

Variables de entorno requeridas:
  AZURE_SPEECH_KEY    → Clave Azure Cognitive Services
  AZURE_SPEECH_REGION → Región (ej: eastus)
"""

import os
import shutil
import sys
import tempfile
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Security, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# Añadir scripts/ al path para importar los módulos
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

app = FastAPI(
    title="Tu Dosis Financiera — Scripts API",
    description="API para generación de videos cortos y carruseles",
    version="1.0.0",
)

# Configurar CORS middleware para permitir requests desde n8n Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, limitar a dominios de n8n
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar API Key authentication
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def get_api_key(authorization: str = Security(api_key_header)):
    """
    Valida la API Key del header Authorization.
    Formato esperado: Authorization: Bearer YOUR_API_KEY
    """
    api_key = os.environ.get("API_KEY")

    # Si no hay API_KEY configurada, permitir sin auth (para testing inicial)
    if not api_key:
        return True

    if not authorization:
        raise HTTPException(status_code=401, detail="API Key requerida en header Authorization")

    # Authorization header formato: "Bearer YOUR_API_KEY"
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Formato de Authorization inválido. Debe ser: Bearer YOUR_API_KEY"
        )

    client_key = authorization[7:]  # Extraer key después de "Bearer "

    if client_key != api_key:
        raise HTTPException(status_code=403, detail="API Key inválida")

    return client_key

# Ajustar FONTS_DIR dentro de los scripts al path correcto del contenedor
# Los scripts usan Path(__file__).parent.parent / "assets" / "fonts"
# Con __file__ = /app/scripts/newsletter_video_gen.py → /app/assets/fonts ✓


# ---------------------------------------------------------------------------
# Modelos de request
# ---------------------------------------------------------------------------

class VideoRequest(BaseModel):
    guion: str
    titulo: str
    hook: Optional[str] = ""
    categoria: Optional[str] = ""
    cta_midroll: Optional[str] = ""
    parte: Optional[str] = ""
    voice: Optional[str] = "es-ES-AlvaroNeural"
    rate: Optional[str] = "+10%"


class CarouselRequest(BaseModel):
    slides_json: dict  # {"slide1": {...}, "slide2": {...}, ...}
    categoria: Optional[str] = "FINANZAS"
    fecha: Optional[str] = None  # YYYY-MM-DD; si None, usa hoy


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health(api_key: str = Security(get_api_key)):
    """Verifica que el servicio está activo y las vars de entorno configuradas."""
    azure_key = os.environ.get("AZURE_SPEECH_KEY", "")
    azure_region = os.environ.get("AZURE_SPEECH_REGION", "")
    return {
        "status": "ok",
        "azure_configured": bool(azure_key and azure_region),
        "azure_region": azure_region or "no configurada",
    }


# ---------------------------------------------------------------------------
# POST /generate-video  (WF05 — Fábrica de Videos Cortos)
# ---------------------------------------------------------------------------

@app.post("/generate-video")
def generate_video(req: VideoRequest, api_key: str = Security(get_api_key)):
    """
    Genera un video MP4 1080x1920 con Azure TTS + MoviePy.
    Devuelve el archivo MP4 directamente como FileResponse.
    n8n recibe el binario y puede subirlo a Google Drive / YouTube.
    """
    import newsletter_video_gen as nvg

    if not req.guion or not req.titulo:
        raise HTTPException(status_code=400, detail="guion y titulo son obligatorios")

    video_id = uuid.uuid4().hex
    output_dir = Path(tempfile.gettempdir()) / f"video_{video_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "video.mp4")
    tmp_audio = str(output_dir / "audio.mp3")

    try:
        print(f"--- [API] Iniciando generación de video: {req.titulo} ---")
        
        # 1. Generar audio con Azure TTS
        print(f"[1/3] Solicitando audio a Azure TTS...")
        nvg.generate_audio_azure(
            text=req.guion,
            voice=req.voice,
            rate=req.rate,
            output_path=tmp_audio,
        )
        print(f"      Audio generado correctamente en {tmp_audio}")

        # 2. Medir duración del audio
        print(f"[2/3] Midiendo duración del audio...")
        from moviepy.editor import AudioFileClip
        audio_check = AudioFileClip(tmp_audio)
        duration = audio_check.duration
        audio_check.close()
        print(f"      Duración detectada: {duration:.2f} segundos")

        # 3. Construir video
        print(f"[3/3] Iniciando montaje del video con MoviePy (esto gasta CPU)...")
        nvg.build_video(
            audio_path=tmp_audio,
            guion_duration=duration,
            titulo=req.titulo,
            hook=req.hook,
            categoria=req.categoria,
            cta_midroll=req.cta_midroll,
            parte=req.parte,
            output_path=output_path,
        )
        print(f"--- [API] Video montado con éxito: {output_path} ---")

    except Exception as e:
        print(f"!!! [API ERROR] Error en el proceso: {str(e)}")
        shutil.rmtree(output_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error generando video: {e}")

    # Devolver el archivo MP4
    # FileResponse envía el binario; n8n lo recibe como Binary en el HTTP Request node
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"video_{video_id}.mp4",
        background=_cleanup_after_response(output_dir),
    )


# ---------------------------------------------------------------------------
# POST /generate-carousel  (WF06 — Fábrica de Carruseles)
# ---------------------------------------------------------------------------

@app.post("/generate-carousel")
def generate_carousel(req: CarouselRequest, api_key: str = Security(get_api_key)):
    """
    Genera 5 imágenes JPG 1080x1080 (Dark Finance Minimal).
    Devuelve JSON con las 5 imágenes en base64 para que n8n las procese
    individualmente y las suba a Google Drive.

    Respuesta:
    {
      "fecha": "2026-02-17",
      "slides": [
        {"num": 1, "filename": "slide_1_2026-02-17.jpg", "base64": "..."},
        ...
      ]
    }
    """
    import base64
    import carousel_gen as cg

    if not req.slides_json:
        raise HTTPException(status_code=400, detail="slides_json es obligatorio")

    fecha = req.fecha or date.today().isoformat()
    carousel_id = uuid.uuid4().hex
    output_dir = Path(tempfile.gettempdir()) / f"carousel_{carousel_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        paths = cg.generate_carousel(
            slides_data=req.slides_json,
            categoria=req.categoria,
            output_dir=str(output_dir / "slides"),
            fecha=fecha,
        )
    except Exception as e:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error generando carrusel: {e}")

    # Convertir cada JPG a base64 para que n8n los maneje sin binarios
    slides = []
    for i, p in enumerate(paths, 1):
        with open(p, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        slides.append({
            "num": i,
            "filename": Path(p).name,
            "base64": img_b64,
        })

    # Limpiar temporales
    shutil.rmtree(output_dir, ignore_errors=True)

    return JSONResponse(content={"fecha": fecha, "slides": slides})


# ---------------------------------------------------------------------------
# Limpieza de temporales tras enviar la respuesta
# ---------------------------------------------------------------------------

def _cleanup_after_response(directory: Path):
    """Devuelve una BackgroundTask de Starlette para borrar el directorio."""
    from starlette.background import BackgroundTask

    def _cleanup():
        shutil.rmtree(directory, ignore_errors=True)

    return BackgroundTask(_cleanup)
