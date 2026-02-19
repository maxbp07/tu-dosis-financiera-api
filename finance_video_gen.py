#!/usr/bin/env python3
"""
finance_video_gen.py — El Mono Financiero
==========================================
Genera un vídeo vertical 9:16 con lógica de Doble Escena:

  Segmento A (contenido): zoom suave 1.0 → 1.10 sobre imagen de main_dir
  Segmento B (CTA):       crash zoom 1.0 → 1.25 sobre imagen de cta_dir

Audio: Azure TTS (AlvaroNeural) — un solo archivo, corte matemático exacto.
Render: 1080x1920 · libx264 · yuv420p · 30fps

Uso:
  python finance_video_gen.py \\
    --text "Tu texto aquí" \\
    --main_dir ./imagenes/main \\
    --cta_dir  ./imagenes/cta \\
    --output   ./output.mp4

Variables de entorno requeridas:
  AZURE_SPEECH_KEY    → Clave de Azure Cognitive Services
  AZURE_SPEECH_REGION → Región del recurso (ej: eastus)
"""

import argparse
import os
import random
import sys
import tempfile
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
VIDEO_W = 1080
VIDEO_H = 1920
VIDEO_FPS = 30
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "medium"
VIDEO_PIX_FMT = "yuv420p"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULT_VOICE = "es-ES-AlvaroNeural"
DEFAULT_RATE = "+10%"
DEFAULT_CTA_DURATION = 4.0


# ---------------------------------------------------------------------------
# Utilidades de imagen
# ---------------------------------------------------------------------------

def list_images(directory: str) -> list[str]:
    """Devuelve lista de imágenes soportadas en el directorio."""
    p = Path(directory)
    if not p.is_dir():
        raise FileNotFoundError(f"Directorio no encontrado: {directory}")
    images = [
        str(f) for f in p.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not images:
        raise ValueError(
            f"No se encontraron imágenes ({', '.join(SUPPORTED_EXTENSIONS)}) en: {directory}"
        )
    return images


def center_crop_clip(clip, target_w: int, target_h: int):
    """
    Redimensiona el clip al tamaño mínimo que cubre target_w × target_h
    manteniendo el ratio de aspecto, luego recorta al centro.
    """
    from moviepy.editor import vfx

    scale = max(target_w / clip.w, target_h / clip.h)
    resized = clip.resize(scale)

    x1 = (resized.w - target_w) / 2
    y1 = (resized.h - target_h) / 2
    cropped = resized.crop(x1=x1, y1=y1, x2=x1 + target_w, y2=y1 + target_h)
    return cropped


def make_zoom_clip(img_path: str, duration: float,
                   scale_start: float, scale_end: float,
                   w: int = VIDEO_W, h: int = VIDEO_H):
    """
    Crea un ImageClip con efecto de zoom lineal.

    Estrategia:
      1. Pre-escalar la imagen al tamaño máximo que alcanzará el zoom (scale_end).
         Esto garantiza que no haya bordes negros en ningún frame.
      2. Aplicar resize dinámico frame a frame.
      3. Hacer crop centrado fijo al tamaño de salida.

    El resultado es un clip de exactamente w × h píxeles durante `duration` segundos.
    """
    from moviepy.editor import ImageClip

    # Pre-escalar al tamaño máximo del zoom + margen de 2% para evitar subpíxeles
    pre_w = int(w * scale_end * 1.02)
    pre_h = int(h * scale_end * 1.02)

    raw = ImageClip(img_path)
    raw = center_crop_clip(raw, pre_w, pre_h)

    # Función de zoom: de scale_start a scale_end de forma lineal
    def zoom_factor(t):
        if duration <= 0:
            return scale_start
        progress = min(t / duration, 1.0)
        return scale_start + (scale_end - scale_start) * progress

    # Aplicar zoom dinámico
    zoomed = raw.resize(zoom_factor)

    # Crop centrado fijo: siempre recortamos el centro de la imagen escalada
    def crop_center(get_frame, t):
        frame = get_frame(t)
        fh, fw = frame.shape[:2]
        x1 = max(0, (fw - w) // 2)
        y1 = max(0, (fh - h) // 2)
        return frame[y1:y1 + h, x1:x1 + w]

    final = zoomed.fl(crop_center, apply_to=["mask"])
    return final.set_duration(duration)


# ---------------------------------------------------------------------------
# Azure TTS
# ---------------------------------------------------------------------------

def generate_audio_azure(text: str, voice: str, rate: str, output_path: str) -> None:
    """
    Genera audio con Azure TTS y lo guarda en output_path (formato MP3).

    Credenciales: variables de entorno AZURE_SPEECH_KEY y AZURE_SPEECH_REGION.
    """
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        raise ImportError(
            "SDK de Azure no encontrado. Ejecuta: pip install azure-cognitiveservices-speech"
        )

    api_key = os.environ.get("AZURE_SPEECH_KEY")
    region = os.environ.get("AZURE_SPEECH_REGION")

    if not api_key:
        raise EnvironmentError(
            "Variable de entorno AZURE_SPEECH_KEY no configurada."
        )
    if not region:
        raise EnvironmentError(
            "Variable de entorno AZURE_SPEECH_REGION no configurada."
        )

    # Construir SSML para controlar velocidad de locución
    ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='es-ES'>
    <voice name='{voice}'>
        <prosody rate='{rate}'>{text}</prosody>
    </voice>
</speak>"""

    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    # MP3 16kHz 32kbps: compatibilidad máxima con moviepy/ffmpeg
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    )

    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"  [Azure TTS] Audio generado: {output_path}")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        raise RuntimeError(
            f"Azure TTS cancelado. Razón: {cancellation.reason}. "
            f"Error: {cancellation.error_details}"
        )
    else:
        raise RuntimeError(f"Azure TTS falló con resultado: {result.reason}")


# ---------------------------------------------------------------------------
# Montaje de vídeo
# ---------------------------------------------------------------------------

def build_video(
    audio_path: str,
    main_img_path: str,
    cta_img_path: str,
    output_path: str,
    cta_duration: float,
) -> None:
    """
    Construye el vídeo final con la lógica de Doble Escena.

    Segmento A: main_img con zoom suave (1.0 → 1.10) durante audio_total - cta_duration
    Segmento B: cta_img con crash zoom (1.0 → 1.25) durante cta_duration
    Audio: un solo archivo, dividido matemáticamente con subclip()
    """
    from moviepy.editor import AudioFileClip, concatenate_videoclips

    print(f"  [Audio] Cargando: {audio_path}")
    audio = AudioFileClip(audio_path)
    total_duration = audio.duration
    seg_a_duration = total_duration - cta_duration

    print(f"  [Tiempos] Total: {total_duration:.2f}s | Segmento A: {seg_a_duration:.2f}s | CTA: {cta_duration:.2f}s")

    if seg_a_duration <= 0:
        raise ValueError(
            f"El audio dura solo {total_duration:.2f}s, que es menor o igual a "
            f"cta_duration ({cta_duration}s). Usa un texto más largo o reduce --cta_duration."
        )

    # --- Segmento A: zoom suave ---
    print(f"  [Segmento A] Imagen: {Path(main_img_path).name}")
    clip_a = make_zoom_clip(
        img_path=main_img_path,
        duration=seg_a_duration,
        scale_start=1.0,
        scale_end=1.10,
    )
    audio_a = audio.subclip(0, seg_a_duration)
    clip_a = clip_a.set_audio(audio_a)

    # --- Segmento B: crash zoom ---
    print(f"  [Segmento B] Imagen: {Path(cta_img_path).name}")
    clip_b = make_zoom_clip(
        img_path=cta_img_path,
        duration=cta_duration,
        scale_start=1.0,
        scale_end=1.25,
    )
    audio_b = audio.subclip(seg_a_duration)
    clip_b = clip_b.set_audio(audio_b)

    # --- Montaje: hard cut ---
    print("  [Montaje] Concatenando clips (hard cut)...")
    final = concatenate_videoclips([clip_a, clip_b], method="compose")

    # --- Render ---
    print(f"  [Render] Escribiendo: {output_path}")
    final.write_videofile(
        output_path,
        fps=VIDEO_FPS,
        codec=VIDEO_CODEC,
        preset=VIDEO_PRESET,
        ffmpeg_params=["-pix_fmt", VIDEO_PIX_FMT],
        audio_codec="aac",
        logger="bar",
    )

    # --- Cerrar recursos ---
    final.close()
    clip_a.close()
    clip_b.close()
    audio.close()
    audio_a.close()
    audio_b.close()

    print(f"  [OK] Vídeo generado: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="El Mono Financiero — Generador de vídeo vertical 9:16 con Doble Escena",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo:
  python finance_video_gen.py \\
    --text "El interés compuesto es la octava maravilla del mundo. ¡Suscríbete y aprende más!" \\
    --main_dir ./imagenes/main \\
    --cta_dir  ./imagenes/cta \\
    --output   ./video_output.mp4

Variables de entorno requeridas:
  AZURE_SPEECH_KEY    — Clave de Azure Cognitive Services
  AZURE_SPEECH_REGION — Región (ej: eastus, westeurope)
        """
    )

    parser.add_argument(
        "--text",
        required=True,
        help="Texto completo a narrar. El CTA debe estar al final."
    )
    parser.add_argument(
        "--main_dir",
        required=True,
        help="Carpeta con imágenes serias/contenido (Segmento A)."
    )
    parser.add_argument(
        "--cta_dir",
        required=True,
        help="Carpeta con imágenes del mono/CTA (Segmento B)."
    )
    parser.add_argument(
        "--output",
        default="output.mp4",
        help="Ruta del MP4 de salida. (default: output.mp4)"
    )
    parser.add_argument(
        "--cta_duration",
        type=float,
        default=DEFAULT_CTA_DURATION,
        help=f"Duración en segundos del Segmento B (Crash Zoom). (default: {DEFAULT_CTA_DURATION})"
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Voz de Azure TTS. (default: {DEFAULT_VOICE})"
    )
    parser.add_argument(
        "--rate",
        default=DEFAULT_RATE,
        help=f"Velocidad de locución Azure SSML. (default: {DEFAULT_RATE})"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 55)
    print(" El Mono Financiero — Generador de Vídeo")
    print("=" * 55)

    # Validar carpetas y seleccionar imágenes aleatorias
    print("\n[1/4] Preparando imágenes...")
    main_images = list_images(args.main_dir)
    cta_images = list_images(args.cta_dir)

    main_img = random.choice(main_images)
    cta_img = random.choice(cta_images)
    print(f"  Segmento A: {Path(main_img).name} ({len(main_images)} disponibles)")
    print(f"  Segmento B: {Path(cta_img).name} ({len(cta_images)} disponibles)")

    # Archivo temporal para el audio
    tmp_audio = os.path.join(tempfile.gettempdir(), f"mono_audio_{uuid.uuid4().hex}.mp3")

    try:
        # Generar audio con Azure TTS
        print("\n[2/4] Generando audio con Azure TTS...")
        generate_audio_azure(
            text=args.text,
            voice=args.voice,
            rate=args.rate,
            output_path=tmp_audio,
        )

        # Construir y renderizar vídeo
        print("\n[3/4] Montando y renderizando vídeo...")
        build_video(
            audio_path=tmp_audio,
            main_img_path=main_img,
            cta_img_path=cta_img,
            output_path=args.output,
            cta_duration=args.cta_duration,
        )

        print("\n[4/4] ¡Listo!")
        print(f"  Vídeo: {os.path.abspath(args.output)}")

    finally:
        # Limpiar temporales siempre, incluso si hay error
        if os.path.exists(tmp_audio):
            os.remove(tmp_audio)
            print(f"  [Limpieza] Temporal eliminado: {tmp_audio}")

    print("=" * 55)


if __name__ == "__main__":
    main()
