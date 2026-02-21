#!/usr/bin/env python3
"""
newsletter_video_gen.py — V2 Multi-Background
==============================================
Genera videos verticales 1080x1920 con:
  - Fondo de video aleatorio (mitad inferior)
  - Personaje animado con lip-sync (mitad superior)
  - Subtitulos virales palabra por palabra
  - Azure TTS con SSML (cheerful + rate 1.15)

Estructura de assets:
  assets/characters/       body.png, character_open.png, character_closed.png, character_angry.png
  assets/videos_fondo/     *.mp4 (fondos aleatorios)
  assets/music/            ambient_lofi_01.mp3 (opcional)

Uso:
  python newsletter_video_gen.py \\
    --guion "Texto del locutor..." \\
    --titulo "Titulo del video" \\
    --output ./output.mp4

Variables de entorno:
  AZURE_SPEECH_KEY    -> Clave de Azure Cognitive Services
  AZURE_SPEECH_REGION -> Region (ej: francecentral)
"""

import argparse
import glob
import math
import os
import random
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Monkey-patch: Pillow 10+ elimino Image.ANTIALIAS, MoviePy 1.0.3 lo necesita
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
VIDEO_W = 1080
VIDEO_H = 1920
VIDEO_FPS = 30
HALF_H = VIDEO_H // 2  # 960px cada mitad

COLOR_BG_TOP = (15, 12, 20)  # Fondo solido oscuro mitad superior
COLOR_SUBTITLE = (255, 215, 0)  # Amarillo #FFD700
COLOR_BLACK = (0, 0, 0)

DEFAULT_VOICE = "es-ES-AlvaroNeural"
BRANDING_TEXT = "TuDosisFinanciera.com"

# Rutas
SCRIPT_DIR = Path(__file__).parent
ASSETS_DIR = SCRIPT_DIR.parent / "assets"
CHARACTERS_DIR = ASSETS_DIR / "characters"
VIDEOS_FONDO_DIR = ASSETS_DIR / "videos_fondo"
BACKGROUNDS_DIR = ASSETS_DIR / "backgrounds"
MUSIC_DIR = ASSETS_DIR / "music"

# Compatibilidad: tambien buscar en "character" (singular) si "characters" no existe
if not CHARACTERS_DIR.exists():
    CHARACTERS_DIR = ASSETS_DIR / "character"

# ---------------------------------------------------------------------------
# Azure TTS con SSML Cheerful
# ---------------------------------------------------------------------------

def generate_audio_azure(text: str, output_path: str, voice: str = "es-ES-AlvaroNeural", rate: str = "1.15") -> None:
    """Genera audio con Azure TTS (AlvaroNeural, cheerful, rate configurable)."""
    print(f"  [Azure TTS] Generando audio para: {text[:50]}...")
    print(f"  [Azure TTS] Voz: {voice} | Rate: {rate}")
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        raise ImportError(
            "SDK de Azure no encontrado. Ejecuta: pip install azure-cognitiveservices-speech"
        )

    api_key = os.environ.get("AZURE_SPEECH_KEY")
    region = os.environ.get("AZURE_SPEECH_REGION")

    if not api_key:
        raise EnvironmentError("Variable AZURE_SPEECH_KEY no configurada.")
    if not region:
        raise EnvironmentError("Variable AZURE_SPEECH_REGION no configurada.")

    ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis'
      xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='es-ES'>
  <voice name='{voice}'>
    <mstts:express-as style='cheerful'>
      <prosody rate='{rate}'>{text}</prosody>
    </mstts:express-as>
  </voice>
</speak>"""

    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
    )
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"  [Azure TTS] Audio generado: {output_path}")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        raise RuntimeError(
            f"Azure TTS cancelado: {cancellation.reason} -- {cancellation.error_details}"
        )
    else:
        raise RuntimeError(f"Azure TTS fallo: {result.reason}")


# ---------------------------------------------------------------------------
# Duracion de audio (sin to_soundarray, compatible NumPy 2.x)
# ---------------------------------------------------------------------------

def get_audio_duration(audio_path: str) -> float:
    """Obtiene duracion del audio usando MoviePy."""
    from moviepy.editor import AudioFileClip
    clip = AudioFileClip(audio_path)
    dur = clip.duration
    clip.close()
    return dur


def extract_audio_amplitudes(audio_path: str, fps: int = 30,
                             window_ms: int = 50) -> np.ndarray:
    """
    Extrae amplitud RMS por frame del audio.
    Usa MoviePy iter_chunks con workaround para NumPy 2.x.
    """
    from moviepy.editor import AudioFileClip

    audio = AudioFileClip(audio_path)
    sample_rate = audio.fps
    duration = audio.duration
    total_frames = int(duration * fps)

    # Workaround NumPy 2.x: recolectar chunks en lista antes de vstack
    chunks = list(audio.iter_chunks(fps=sample_rate, quantize=True,
                                     nbytes=2, chunksize=2000))
    samples = np.vstack(chunks) if chunks else np.array([])
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    audio.close()

    window_samples = int(sample_rate * window_ms / 1000)
    amplitudes = np.zeros(total_frames, dtype=np.float32)

    for frame_idx in range(total_frames):
        t_center = (frame_idx + 0.5) / fps
        sample_center = int(t_center * sample_rate)
        start = max(0, sample_center - window_samples // 2)
        end = min(len(samples), sample_center + window_samples // 2)

        if end > start:
            window = samples[start:end]
            amplitudes[frame_idx] = np.sqrt(np.mean(window ** 2))

    # Normalizar con percentil 95
    p95 = np.percentile(amplitudes, 95) if len(amplitudes) > 0 else 0
    if p95 > 0:
        amplitudes = np.clip(amplitudes / p95, 0.0, 1.0)

    return amplitudes


# ---------------------------------------------------------------------------
# Sistema de Fondo Aleatorio
# ---------------------------------------------------------------------------

def scan_background_videos(folder: Path) -> list:
    """Escanea carpeta y devuelve lista de paths .mp4."""
    pattern = str(folder / "*.mp4")
    videos = glob.glob(pattern)
    if not videos:
        raise FileNotFoundError(
            f"No se encontraron videos .mp4 en: {folder}\n"
            f"Coloca al menos un video de fondo en assets/videos_fondo/"
        )
    print(f"  [Fondos] Encontrados: {len(videos)} videos")
    for v in videos:
        print(f"    - {Path(v).name}")
    return videos


def prepare_background_clip(video_path: str, target_duration: float,
                            target_w: int = VIDEO_W, target_h: int = HALF_H):
    """
    Prepara clip de fondo: smart crop temporal + resize/crop espacial con mejor calidad.
    Devuelve VideoClip de tamanyo target_w x target_h con duracion target_duration.
    """
    from moviepy.editor import VideoFileClip, vfx

    print(f"  [Fondo] Seleccionado: {Path(video_path).name}")
    clip = VideoFileClip(video_path)
    print(f"  [Fondo] Original: {clip.size[0]}x{clip.size[1]}, {clip.duration:.1f}s")

    # --- Smart Crop Temporal ---
    if clip.duration > target_duration:
        max_start = clip.duration - target_duration
        start_time = random.uniform(0, max_start)
        clip = clip.subclip(start_time, start_time + target_duration)
        print(f"  [Fondo] Recorte temporal: {start_time:.1f}s -> {start_time + target_duration:.1f}s")
    elif clip.duration < target_duration:
        clip = clip.fx(vfx.loop, duration=target_duration)
        print(f"  [Fondo] Loop aplicado para cubrir {target_duration:.1f}s")

    # --- Smart Crop Espacial con mejor calidad ---
    src_w, src_h = clip.size
    # Calcular escala para cubrir todo el target (sin bordes negros)
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    # Usar método de redimensionamiento de alta calidad con PIL
    from PIL import Image

    def resize_frame(get_frame, t):
        frame = get_frame(t)
        img = Image.fromarray(frame)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        return np.array(img)

    clip = clip.fl(resize_frame)

    # Center crop al tamanyo final
    x_offset = (new_w - target_w) // 2
    y_offset = (new_h - target_h) // 2
    clip = clip.crop(x1=x_offset, y1=y_offset,
                     x2=x_offset + target_w, y2=y_offset + target_h)

    print(f"  [Fondo] Final: {target_w}x{target_h}")
    return clip


# ---------------------------------------------------------------------------
# Carga de sprites del personaje
# ---------------------------------------------------------------------------

def load_character_sprites(char_dir: Path) -> dict:
    """Carga los 4 sprites del personaje como PIL Images."""
    sprites = {}
    sprite_files = {
        "body": "body.png",
        "open": "character_open.png",
        "closed": "character_closed.png",
        "angry": "character_angry.png",
    }

    for name, filename in sprite_files.items():
        path = char_dir / filename
        if path.exists():
            sprites[name] = Image.open(str(path)).convert("RGBA")
            print(f"  [Character] {name}: {path.name} ({sprites[name].size})")
        else:
            print(f"  [Character] AVISO: {filename} no encontrado")
            sprites[name] = None

    if sprites["body"] is None:
        raise FileNotFoundError(f"body.png es obligatorio en {char_dir}")

    # Fallback: si falta algun sprite, usar body
    for key in ["open", "closed", "angry"]:
        if sprites[key] is None:
            sprites[key] = sprites["body"]

    return sprites


# ---------------------------------------------------------------------------
# Deteccion de ira en el guion
# ---------------------------------------------------------------------------

def detect_angry_segments(guion: str, words_per_segment: int = 3) -> list:
    """
    Analiza el guion y marca segmentos como 'angry' si contienen
    '!' o si >60% de los caracteres son mayusculas.
    Devuelve lista de bools, una por segmento.
    """
    words = guion.split()
    segments = []
    for i in range(0, len(words), words_per_segment):
        chunk = " ".join(words[i:i + words_per_segment])
        segments.append(chunk)

    angry_flags = []
    for seg in segments:
        has_exclamation = "!" in seg or "\u00a1" in seg  # ! o  inverted !
        alpha_chars = [c for c in seg if c.isalpha()]
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
        is_angry = has_exclamation or upper_ratio > 0.6
        angry_flags.append(is_angry)

    return angry_flags


# ---------------------------------------------------------------------------
# Schedule de parpadeo
# ---------------------------------------------------------------------------

def generate_blink_schedule(duration: float, fps: int,
                            min_interval: float = 2.0,
                            max_interval: float = 5.0,
                            blink_duration: float = 0.15) -> list:
    """Pre-calcula estado de ojos por frame: True = parpadeando."""
    total_frames = int(duration * fps)
    blink_frames = [False] * total_frames
    blink_len = max(2, int(blink_duration * fps))

    t = random.uniform(min_interval, max_interval)
    while t < duration:
        center_frame = int(t * fps)
        half = blink_len // 2
        for offset in range(-half, half + 1):
            f = center_frame + offset
            if 0 <= f < total_frames:
                blink_frames[f] = True
        t += random.uniform(min_interval, max_interval)

    return blink_frames


# ---------------------------------------------------------------------------
# Renderizado del personaje frame-by-frame como ImageClip
# ---------------------------------------------------------------------------

def load_upper_background(target_w: int = VIDEO_W, target_h: int = HALF_H) -> Image.Image:
    """Carga un fondo aleatorio de la carpeta backgrounds y lo recorta a la mitad superior."""
    # Buscar todos los archivos PNG/JPG en la carpeta backgrounds
    bg_files = list(BACKGROUNDS_DIR.glob("*.png")) + list(BACKGROUNDS_DIR.glob("*.jpg"))

    if bg_files:
        # Seleccionar aleatoriamente uno de los fondos
        bg_path = random.choice(bg_files)
        bg = Image.open(str(bg_path)).convert("RGB")
        # Resize al ancho del video manteniendo proporcion
        ratio = target_w / bg.width
        new_h = int(bg.height * ratio)
        bg = bg.resize((target_w, new_h), Image.LANCZOS)
        # Recortar la parte superior (mitad superior del fondo)
        if new_h > target_h:
            bg = bg.crop((0, 0, target_w, target_h))
        elif new_h < target_h:
            # Si es mas pequeno, centrar sobre fondo oscuro
            canvas = Image.new("RGB", (target_w, target_h), COLOR_BG_TOP)
            y_off = (target_h - new_h) // 2
            canvas.paste(bg, (0, y_off))
            bg = canvas
        print(f"  [Background] Fondo superior: {bg_path.name} ({target_w}x{target_h})")
        return bg
    else:
        print(f"  [Background] No se encontraron fondos. Usando color solido.")
        return Image.new("RGB", (target_w, target_h), COLOR_BG_TOP)


def build_character_clip(sprites: dict, amplitudes: np.ndarray,
                         blink_schedule: list, angry_schedule: list,
                         duration: float, fps: int,
                         bg_image: Image.Image = None,
                         target_w: int = VIDEO_W, target_h: int = HALF_H):
    """
    Construye un VideoClip de la mitad superior con el personaje animado.
    Fondo de imagen (podcast_studio) + sprite centrado.
    """
    from moviepy.editor import VideoClip

    # Cargar fondo si no se paso uno
    if bg_image is None:
        bg_image = load_upper_background(target_w, target_h)

    # Escalar sprites para que quepan en la mitad superior
    body = sprites["body"]
    # Calcular escala para que el personaje ocupe ~80% del alto disponible
    max_char_h = int(target_h * 0.85)
    scale = min(max_char_h / body.height, (target_w * 0.9) / body.width)
    if scale < 1.0 or scale > 2.5:
        scale = min(max(scale, 0.5), 2.5)

    scaled_sprites = {}
    for name, img in sprites.items():
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        scaled_sprites[name] = img.resize((new_w, new_h), Image.LANCZOS)

    print(f"  [Character] Escala: {scale:.2f}x -> {scaled_sprites['body'].size}")

    char_w = scaled_sprites["body"].width
    char_h = scaled_sprites["body"].height
    char_x = (target_w - char_w) // 2
    char_y = (target_h - char_h) // 2 + 30  # ligeramente hacia abajo

    # Umbral RMS para boca abierta (equivalente a -25dB normalizado)
    rms_threshold = 0.15

    total_frames = int(duration * fps)

    # Pre-calcular cuantos segmentos de ira hay
    n_angry = len(angry_schedule)

    def make_frame(t):
        frame_idx = min(int(t * fps), total_frames - 1)

        # Usar fondo de imagen (podcast studio)
        frame = bg_image.copy()

        # Determinar estado del personaje
        is_blinking = blink_schedule[frame_idx] if frame_idx < len(blink_schedule) else False
        rms = amplitudes[frame_idx] if frame_idx < len(amplitudes) else 0.0
        is_speaking = rms > rms_threshold

        # Determinar si estamos en segmento angry
        if n_angry > 0:
            segment_idx = min(int(frame_idx / total_frames * n_angry), n_angry - 1)
            is_angry = angry_schedule[segment_idx]
        else:
            is_angry = False

        # Seleccionar sprite
        if is_blinking:
            sprite = scaled_sprites["closed"]
        elif is_angry and is_speaking:
            sprite = scaled_sprites["angry"]
        elif is_speaking:
            sprite = scaled_sprites["open"]
        else:
            sprite = scaled_sprites["body"]

        # Respiracion sutil
        breath_offset = int(3 * math.sin(2 * math.pi * t / 3.5))
        paste_y = char_y + breath_offset

        # Pegar sprite
        frame.paste(sprite, (char_x, paste_y), sprite)

        return np.array(frame)

    clip = VideoClip(make_frame, duration=duration)
    return clip


# ---------------------------------------------------------------------------
# Subtitulos virales
# ---------------------------------------------------------------------------

def _get_subtitle_font(size: int):
    """Intenta cargar Impact/Arial Bold; fallback a default de Pillow."""
    for font_name in ["impact.ttf", "Impact.ttf", "arialbd.ttf", "Arial Bold.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except (OSError, IOError):
            pass
    # Buscar en carpeta de Windows
    win_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for font_name in ["impact.ttf", "arialbd.ttf", "arial.ttf"]:
        font_path = win_fonts / font_name
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size)
            except (OSError, IOError):
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _render_subtitle_image(text: str, font, max_w: int,
                           color=(255, 215, 0), stroke_color=(0, 0, 0),
                           stroke_width: int = 3) -> Image.Image:
    """Renderiza texto con borde negro sobre fondo transparente usando Pillow."""
    # Calcular tamano del texto
    tmp = Image.new("RGBA", (max_w, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)

    # Wrap text si es necesario
    words = text.split()
    lines = []
    current = []
    for word in words:
        test = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_w - 20 and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))

    # Calcular alto total
    line_h = draw.textbbox((0, 0), "Ag", font=font)[3] + 5
    total_h = line_h * len(lines) + stroke_width * 2 + 10
    total_w = max_w

    img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    y = stroke_width + 5
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        x = (total_w - line_w) // 2

        # Dibujar stroke (borde negro) con offsets
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if dx * dx + dy * dy <= stroke_width * stroke_width:
                    draw.text((x + dx, y + dy), line, font=font,
                              fill=(*stroke_color, 255))

        # Dibujar texto principal
        draw.text((x, y), line, font=font, fill=(*color, 255))
        y += line_h

    return img


def build_subtitle_clips(guion: str, duration: float,
                         target_w: int = VIDEO_W, subtitle_y: int = HALF_H - 40):
    """
    Genera subtitulos virales usando Pillow (sin ImageMagick).
    Palabra por palabra o frases cortas, centrados en zona de union.
    """
    from moviepy.editor import ImageClip

    font = _get_subtitle_font(85)

    words = guion.split()
    group_size = 2
    segments = []
    for i in range(0, len(words), group_size):
        chunk = " ".join(words[i:i + group_size])
        segments.append(chunk.upper())

    if not segments:
        return None

    segment_duration = duration / len(segments)
    clips = []

    for i, text in enumerate(segments):
        start_time = i * segment_duration

        # Renderizar con Pillow
        img = _render_subtitle_image(text, font, target_w - 100)
        img_array = np.array(img)

        txt_clip = (ImageClip(img_array, ismask=False, transparent=True)
                    .set_start(start_time)
                    .set_duration(segment_duration)
                    .set_position(("center", subtitle_y)))

        clips.append(txt_clip)

    print(f"  [Subtitulos] {len(segments)} segmentos, {segment_duration:.2f}s cada uno")
    return clips


def _render_branding_image(text: str, color=(218, 165, 32)) -> Image.Image:
    """Renderiza branding como imagen PIL."""
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except (OSError, IOError):
        try:
            font = ImageFont.load_default(size=28)
        except TypeError:
            font = ImageFont.load_default()

    tmp = Image.new("RGBA", (VIDEO_W, 60), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    img = Image.new("RGBA", (text_w + 20, text_h + 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((10, 5), text, font=font, fill=(*color, 255))
    return img


# ---------------------------------------------------------------------------
# Mezcla de audio (narracion + musica)
# ---------------------------------------------------------------------------

def mix_audio(narration_path: str, duration: float):
    """Mezcla narracion con musica de fondo (si existe)."""
    from moviepy.editor import AudioFileClip, CompositeAudioClip

    narration = AudioFileClip(narration_path)

    # Buscar musica
    music_files = list(MUSIC_DIR.glob("*.mp3"))
    if not music_files:
        print("  [Audio] Sin musica de fondo.")
        return narration

    track_path = random.choice(music_files)
    print(f"  [Audio] Musica: {track_path.name}")

    try:
        music = AudioFileClip(str(track_path))
        music = music.volumex(0.12)

        if music.duration < duration:
            from moviepy.audio.fx.audio_loop import audio_loop
            music = audio_loop(music, duration=duration)
        else:
            music = music.subclip(0, duration)

        music = music.audio_fadein(2.0).audio_fadeout(3.0)
        return CompositeAudioClip([narration, music])
    except Exception as e:
        print(f"  [Audio] Error con musica: {e}. Usando solo narracion.")
        return narration


# ---------------------------------------------------------------------------
# Motor principal de composicion
# ---------------------------------------------------------------------------

def build_video(audio_path: str, duration: float, guion: str,
                titulo: str, output_path: str) -> None:
    """
    Compone el video final:
      - Mitad superior: personaje animado sobre fondo solido
      - Mitad inferior: video de fondo aleatorio
      - Subtitulos en la zona central
      - Audio: narracion + musica
    """
    from moviepy.editor import (
        CompositeVideoClip, ColorClip, concatenate_videoclips
    )

    print("\n  === COMPOSICION DE VIDEO ===")

    # --- 1. Fondo de video aleatorio (mitad inferior) ---
    print("\n  [1/5] Preparando fondo de video...")
    bg_videos = scan_background_videos(VIDEOS_FONDO_DIR)
    selected_bg = random.choice(bg_videos)
    bg_clip = prepare_background_clip(selected_bg, duration)
    bg_clip = bg_clip.set_position((0, HALF_H))

    # --- 2. Cargar personaje ---
    print("\n  [2/5] Cargando personaje...")
    sprites = load_character_sprites(CHARACTERS_DIR)

    # --- 3. Analisis de audio (lip sync) ---
    print("\n  [3/5] Analizando audio para lip-sync...")
    amplitudes = extract_audio_amplitudes(audio_path, VIDEO_FPS)
    blink_schedule = generate_blink_schedule(duration, VIDEO_FPS)
    angry_schedule = detect_angry_segments(guion)
    print(f"  [Lip Sync] {len(amplitudes)} frames | "
          f"Parpadeos programados | "
          f"{sum(angry_schedule)}/{len(angry_schedule)} segmentos angry")

    # --- 4. Clip del personaje (mitad superior con fondo podcast) ---
    print("\n  [4/5] Cargando fondo + renderizando personaje animado...")
    upper_bg = load_upper_background()
    char_clip = build_character_clip(
        sprites, amplitudes, blink_schedule, angry_schedule,
        duration, VIDEO_FPS, bg_image=upper_bg
    )
    char_clip = char_clip.set_position((0, 0))

    # --- 5. Subtitulos ---
    print("\n  [5/5] Generando subtitulos virales...")
    subtitle_clips = build_subtitle_clips(guion, duration)

    # --- Composicion final ---
    print("\n  [Render] Componiendo video final...")

    # Fondo negro completo como base
    base = ColorClip(size=(VIDEO_W, VIDEO_H), color=COLOR_BG_TOP).set_duration(duration)

    layers = [base, char_clip, bg_clip]
    if subtitle_clips:
        layers.extend(subtitle_clips)

    # Branding (esquina inferior, renderizado con Pillow)
    try:
        from moviepy.editor import ImageClip as _IC
        branding_img = _render_branding_image(BRANDING_TEXT)
        branding_arr = np.array(branding_img)
        branding = (_IC(branding_arr, ismask=False, transparent=True)
                    .set_duration(duration)
                    .set_position(("center", VIDEO_H - 60)))
        layers.append(branding)
    except Exception:
        pass

    video = CompositeVideoClip(layers, size=(VIDEO_W, VIDEO_H))

    # --- Audio ---
    print("  [Audio] Mezclando audio...")
    final_audio = mix_audio(audio_path, duration)
    video = video.set_audio(final_audio)

    # --- Codificar ---
    print(f"  [Render] Codificando: {output_path}")
    video.write_videofile(
        output_path,
        fps=VIDEO_FPS,
        codec="libx264",
        preset="slow",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "18"],
        audio_codec="aac",
        audio_bitrate="192k",
        logger="bar",
    )

    video.close()
    bg_clip.close()
    print(f"  [OK] Video generado: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Newsletter Video Gen V2 - Multi-Background + Lip-Sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo:
  python newsletter_video_gen.py \\
    --guion "El 80 por ciento de los espanoles no tiene fondo de emergencia..." \\
    --titulo "Fondo de Emergencia" \\
    --output ./video_tip_001.mp4

Assets requeridos:
  assets/characters/          body.png, character_open.png, character_closed.png, character_angry.png
  assets/videos_fondo/        *.mp4 (minimo 1 video de fondo)
  assets/music/               *.mp3 (opcional, musica de fondo)
        """
    )

    parser.add_argument("--guion", required=True,
                        help="Texto completo del guion.")
    parser.add_argument("--titulo", required=True,
                        help="Titulo del video.")
    parser.add_argument("--output", default="output.mp4",
                        help="Ruta del MP4 de salida. (default: output.mp4)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print(" Newsletter Video Gen V2 - Multi-Background")
    print("=" * 60)
    print(f"  Titulo: {args.titulo}")
    print(f"  Guion:  {args.guion[:80]}...")

    tmp_audio = os.path.join(
        tempfile.gettempdir(),
        f"newsletter_audio_{uuid.uuid4().hex}.mp3"
    )

    try:
        # 1. Generar audio
        print("\n[1/3] Generando audio con Azure TTS (cheerful, rate 1.15)...")
        generate_audio_azure(text=args.guion, output_path=tmp_audio)

        # 2. Medir duracion
        duration = get_audio_duration(tmp_audio)
        print(f"  Duracion del audio: {duration:.2f}s")

        # 3. Construir video
        print("\n[2/3] Construyendo video...")
        build_video(
            audio_path=tmp_audio,
            duration=duration,
            guion=args.guion,
            titulo=args.titulo,
            output_path=args.output,
        )

        print(f"\n[3/3] Listo!")
        print(f"  Video: {os.path.abspath(args.output)}")
        print(f"OUTPUT_PATH:{os.path.abspath(args.output)}")

    finally:
        if os.path.exists(tmp_audio):
            try:
                os.remove(tmp_audio)
                print(f"  [Limpieza] Temporal eliminado.")
            except PermissionError:
                print(f"  [Limpieza] No se pudo eliminar temporal (en uso).")

    print("=" * 60)


if __name__ == "__main__":
    main()
