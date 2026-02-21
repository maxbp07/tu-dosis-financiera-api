# =============================================================================
# Dockerfile — tudosis-scripts
# Servicio Python con FFmpeg + FastAPI para Tu Dosis Financiera
# Expone: POST /generate-video | POST /generate-carousel | GET /health
# =============================================================================

FROM python:3.11-slim

# Instalar dependencias del sistema: ffmpeg + libGL (Pillow/moviepy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements primero (cache de Docker)
COPY scripts/requirements.txt ./requirements.txt

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        fastapi==0.111.0 \
        uvicorn[standard]==0.29.0 \
        python-multipart==0.0.9 \
    && pip install --no-cache-dir -r requirements.txt

# Copiar scripts
COPY scripts/ ./scripts/

# Copiar app.py (API FastAPI)
COPY app.py ./app.py

# Crear carpetas de trabajo y assets
RUN mkdir -p /app/assets/fonts /app/assets/character /app/assets/backgrounds \
             /app/assets/music /app/outputs/videos /app/outputs/carruseles

# Descargar fuentes tipográficas (Montserrat Bold + Inter)
RUN curl -sL "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf" \
         -o /app/assets/fonts/Montserrat-Bold.ttf \
    && curl -sL "https://github.com/rsms/inter/raw/master/docs/font-files/Inter-Regular.otf" \
         -o /app/assets/fonts/Inter-Regular.ttf \
    && curl -sL "https://github.com/rsms/inter/raw/master/docs/font-files/Inter-SemiBold.otf" \
         -o /app/assets/fonts/Inter-SemiBold.ttf

# Copiar assets del personaje animado, fondos y musica
COPY character_angry.png /app/assets/character/
COPY character_closed.png /app/assets/character/
COPY character_open.png /app/assets/character/
COPY body.png /app/assets/character/
COPY Gemini_Generated_Image_4hsu3s4hsu3s4hsu.png /app/assets/backgrounds/
COPY Gemini_Generated_Image_sl5zybsl5zybsl5z.png /app/assets/backgrounds/
COPY ambient_lofi_01.mp3 /app/assets/music/

# FIX: Puerto 80 (Easypanel proxy apunta a puerto 80)
EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
