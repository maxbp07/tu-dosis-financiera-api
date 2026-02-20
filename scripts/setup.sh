#!/usr/bin/env bash
# =============================================================================
# setup.sh — Instalación del entorno completo
# Proyecto: Tu Dosis Financiera + El Mono Financiero
# Entorno: VPS Linux (Debian/Ubuntu — Easypanel en Hostinger)
# =============================================================================
set -e

echo "========================================="
echo " Tu Dosis Financiera — Setup Completo"
echo "========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FONTS_DIR="$PROJECT_DIR/assets/fonts"

# ---------------------------------------------------------------------------
# 1. Dependencias del sistema
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Instalando dependencias del sistema..."
sudo apt-get update -qq
sudo apt-get install -y ffmpeg python3-venv python3-pip unzip curl

echo "      ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
echo "      python: $(python3 --version)"

# ---------------------------------------------------------------------------
# 2. Entorno virtual Python
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Creando entorno virtual Python..."
python3 -m venv "$PROJECT_DIR/venv"
source "$PROJECT_DIR/venv/bin/activate"

pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt"
echo "      Dependencias Python instaladas."

# ---------------------------------------------------------------------------
# 3. Fuentes tipográficas (para carousel_gen.py y newsletter_video_gen.py)
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Descargando fuentes (Montserrat + Inter)..."
mkdir -p "$FONTS_DIR"

# Montserrat Bold
if [ ! -f "$FONTS_DIR/Montserrat-Bold.ttf" ]; then
    echo "      Descargando Montserrat..."
    curl -sL "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf" \
         -o "$FONTS_DIR/Montserrat-Bold.ttf"
    echo "      Montserrat-Bold.ttf descargada."
else
    echo "      Montserrat-Bold.ttf ya existe, omitiendo."
fi

# Inter Regular
if [ ! -f "$FONTS_DIR/Inter-Regular.ttf" ]; then
    echo "      Descargando Inter Regular..."
    curl -sL "https://github.com/rsms/inter/raw/master/docs/font-files/Inter-Regular.otf" \
         -o "$FONTS_DIR/Inter-Regular.ttf"
    echo "      Inter-Regular.ttf descargada."
else
    echo "      Inter-Regular.ttf ya existe, omitiendo."
fi

# Inter SemiBold
if [ ! -f "$FONTS_DIR/Inter-SemiBold.ttf" ]; then
    echo "      Descargando Inter SemiBold..."
    curl -sL "https://github.com/rsms/inter/raw/master/docs/font-files/Inter-SemiBold.otf" \
         -o "$FONTS_DIR/Inter-SemiBold.ttf"
    echo "      Inter-SemiBold.ttf descargada."
else
    echo "      Inter-SemiBold.ttf ya existe, omitiendo."
fi

echo "      Fuentes listas en: $FONTS_DIR"

# ---------------------------------------------------------------------------
# 4. Estructura de carpetas del proyecto
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Creando estructura de carpetas..."
mkdir -p "$PROJECT_DIR/assets/fonts"
mkdir -p "$PROJECT_DIR/outputs/videos"
mkdir -p "$PROJECT_DIR/outputs/carruseles"
mkdir -p "$PROJECT_DIR/outputs/mono-financiero"
echo "      Carpetas creadas."

# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
echo ""
echo "========================================="
echo " Instalacion completa"
echo "========================================="
echo ""
echo "OBLIGATORIO — Configura en Easypanel -> Environment Variables:"
echo "  AZURE_SPEECH_KEY=<tu_clave_de_azure>"
echo "  AZURE_SPEECH_REGION=<tu_region>   (ej: eastus, westeurope)"
echo ""
echo "Comandos de prueba:"
echo ""
echo "  # Activar entorno:"
echo "  source $PROJECT_DIR/venv/bin/activate"
echo ""
echo "  # Test carrusel (no requiere Azure):"
echo "  python $SCRIPT_DIR/carousel_gen.py \\"
echo "    --slides_json '{\"slide1\":{\"titulo\":\"PRUEBA\",\"subtitulo\":\"Funciona\"}}' \\"
echo "    --categoria \"AHORRO\" \\"
echo "    --output_dir $PROJECT_DIR/outputs/carruseles \\"
echo "    --fecha \$(date +%Y-%m-%d)"
echo ""
echo "  # Test video newsletter (requiere Azure):"
echo "  python $SCRIPT_DIR/newsletter_video_gen.py \\"
echo "    --guion \"El interés compuesto es la herramienta más poderosa del inversor.\" \\"
echo "    --titulo \"Interés Compuesto\" \\"
echo "    --hook \"¿Lo estás usando?\" \\"
echo "    --categoria \"INVERSION\" \\"
echo "    --output $PROJECT_DIR/outputs/videos/test.mp4"
echo ""
echo "  # Test video El Mono Financiero (requiere Azure + imágenes):"
echo "  python $SCRIPT_DIR/finance_video_gen.py \\"
echo "    --text \"El ahorro automático es tu mejor aliado. Suscribete!\" \\"
echo "    --main_dir /ruta/imagenes/main \\"
echo "    --cta_dir /ruta/imagenes/cta \\"
echo "    --output $PROJECT_DIR/outputs/mono-financiero/test.mp4"
echo ""
