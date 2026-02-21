import os
import uuid
import shutil
import tempfile
from datetime import date
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import sys

# Añadir scripts/ al path
sys.path.append(str(Path(__file__).parent / "scripts"))

app = FastAPI(title="Tu Dosis Financiera API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("API_KEY", "Maximussuper-4")
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(header: str = Depends(api_key_header)):
    if header == f"Bearer {API_KEY}" or header == API_KEY:
        return header
    raise HTTPException(status_code=401, detail="API Key no válida")

class VideoRequest(BaseModel):
    guion: str
    titulo: str
    hook: str = ""
    categoria: str = "Finanzas"
    voice: str = "es-ES-AlvaroNeural"
    rate: str = "1.15"
    cta_midroll: bool = False
    parte: int = 1

def _cleanup_after_response(path: Path):
    def cleanup():
        shutil.rmtree(path, ignore_errors=True)
    return cleanup

@app.get("/health")
def health():
    # Publico para que Easypanel no de 401
    return {"status": "ok", "date": date.today().isoformat()}

@app.post("/generate-video")
def generate_video(req: VideoRequest, api_key: str = Security(get_api_key)):
    import newsletter_video_gen as nvg
    video_id = uuid.uuid4().hex
    output_dir = Path(tempfile.gettempdir()) / f"video_{video_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "video.mp4")
    tmp_audio = str(output_dir / "audio.mp3")

    try:
        print(f"--- [API] Iniciando: {req.titulo} ---")
        nvg.generate_audio_azure(text=req.guion, voice=req.voice, rate=req.rate, output_path=tmp_audio)
        dur = nvg.get_audio_duration(tmp_audio)
        nvg.build_video(audio_path=tmp_audio, duration=dur, guion=req.guion, titulo=req.titulo, output_path=output_path)
    except Exception as e:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

    return FileResponse(path=output_path, media_type="video/mp4", background=_cleanup_after_response(output_dir))
