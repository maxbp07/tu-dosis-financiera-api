#!/usr/bin/env python3
"""
phytom_video_gen.py — Video de Phytom Animado
==============================================
Genera videos verticales 1080x1920 con:
  - Phytom animado con lip-sync (mitad superior)
  - Fondo espacial (mitad inferior)
  - Subtítulos personalizados
  - Azure TTS con voz de personaje (LauraNeural, cheerful, rate 1.20)

Estructura de assets:
  assets/character/       body.png, character_open.png, character_closed.png, character_angry.png
  assets/videos_fondo/     *.mp4 (fondos espaciales o existentes)
  assets/music/            *.mp3 (música espacial opcional)

Uso:
  python phytom_video_gen.py \
    --guion "¡Hola! Soy Phytom, tu amigo alienígena..." \
    --titulo "Phytom Te Saluda" \
    --output ./phytom_video.mp4

Variables de entorno:
  AZURE_SPEECH_KEY    -> Clave de Azure Cognitive Services
  AZURE_SPEECH_REGION -> Region (ej: francecentral)
  PHYTOM_VOICE       -> Voz para Phytom (default: es-ES-LauraNeural)
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

COLOR_BG_TOP = (10, 10, 30)  # Fondo oscuro espacial (azul/noche)
COLOR_SUBTITLE = (100, 255, 255)  # Cian brillante para estilo alienígena
COLOR_BLACK = (0, 0, 0)
COLOR_PHYTOM_GLOW = (150, 100, 255)  # Brillo característico de Phytom

DEFAULT_PHYTOM_VOICE = "es-ES-LauraNeural"
PHYTOM_BRANDING = "PhytomVerse"
PHYTOM_BADGE = "👽 ALIENígena"

# Rutas
SCRIPT_DIR = Path(__file__).parent
ASSETS_DIR = SCRIPT_DIR.parent / "assets"
CHARACTERS_DIR = ASSETS_DIR / "character"
VIDEOS_FONDO_DIR = ASSETS_DIR / "videos_fondo"
BACKGROUNDS_DIR = ASSETS_DIR / "backgrounds"
MUSIC_DIR = ASSETS_DIR / "music"

# Compatibilidad: tambien buscar en "characters" (plural) si "character" no existe
if not CHARACTERS_DIR.exists():
    CHARACTERS_DIR = ASSETS_DIR / "characters"

# ---------------------------------------------------------------------------
# Azure TTS con SSML para voz de Phytom
# ---------------------------------------------------------------------------

def get_phytom_voice():
    """Obtiene la configuración de voz para Phytom."""
    return os.environ.get("PHYTOM_VOICE", DEFAULT_PHYTOM_VOICE)

def generate_audio_phytom(text: str, output_path: str) -> None:
    """Genera audio con Azure TTS (LauraNeural, cheerful, rate 1.20)."""
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        raise ImportError(
            "SDK de Azure no encontrado. Ejecuta: pip install azure-cognitiveservices-speech"
        )

    api_key = os.environ.get("AZURE_SPEECH_KEY")
    region = os.environ.get("AZURE_SPEECH_REGION")
    voice_name = get_phytom_voice()

    if not api_key:
        raise EnvironmentError("Variable AZURE_SPEECH_KEY no configurada.")
    if not region:
        raise EnvironmentError("Variable AZURE_SPEECH_REGION no configurada.")

    ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis'
      xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='es-ES'>
  <voice name='{voice_name}'>
    <mstts:express-as style='cheerful'>
      <prosody rate='1.20'>{text}</prosody>
    </mstts:express-as>
  </voice>
</speak>"""

    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio48KHz192KBitRateMonoMp3
    )
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"  [Azure TTS Phytom] Audio generado: {output_path}")
        print(f"  [Voz] Usando: {voice_name} (rate 1.20)")
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
# Sistema de Fondo Aleatorio con preferencia espacial
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
    Prepara clip de fondo: smart crop temporal + resize/crop espacial.
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

    # --- Smart Crop Espacial (resize + center crop) ---
    src_w, src_h = clip.size
    # Calcular escala para cubrir todo el target (sin bordes negros)
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    clip = clip.resize((new_w, new_h))

    # Center crop al tamanyo final
    x_offset = (new_w - target_w) // 2
    y_offset = (new_h - target_h) // 2
    clip = clip.crop(x1=x_offset, y1=y_offset,
                     x2=x_offset + target_w, y2=y_offset + target_h)

    print(f"  [Fondo] Final: {target_w}x{target_h}")
    return clip

# ---------------------------------------------------------------------------
# Carga de sprites de Phytom
# ---------------------------------------------------------------------------

def load_phytom_sprites(char_dir: Path) -> dict:
    """Carga los 4 sprites de Phytom como PIL Images."""
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
            print(f"  [Phytom] {name}: {path.name} ({sprites[name].size})")
        else:
            print(f"  [Phytom] AVISO: {filename} no encontrado")

    if sprites["body"] is None:
        raise FileNotFoundError(f"body.png es obligatorio en {char_dir}")

    # Fallback: si falta algun sprite, usar body
    for key in ["open", "closed", "angry"]:
        if sprites[key] is None:
            sprites[key] = sprites["body"]

    return sprites

# ---------------------------------------------------------------------------
# Estados emocionales de Phytom
# ---------------------------------------------------------------------------

def detect_phytom_emotions(guion: str, words_per_segment: int = 3) -> list:
    """
    Detecta emociones específicas de Phytom basado en palabras clave.
    Devuelve lista de emociones: ['normal', 'excited', 'curious', 'scared']
    """
    words = guion.lower()
    emotions = []

    # Palabras clave para cada emoción
    excited_keywords = ["!", "wow", "increíble", "genial", "fantástico", "emocionado", "á"]
    curious_keywords = ["¿", "?", "cómo", "qué", "por qué", "interesante", "curioso"]
    scared_keywords = ["ay", "ayuda", "peligro", "miedo", "asustado", "oh no"]

    # Dividir en segmentos
    segment_words = guion.split()
    segments = []
    for i in range(0, len(segment_words), words_per_segment):
        chunk = " ".join(segment_words[i:i + words_per_segment])
        segments.append(chunk.lower())

    for segment in segments:
        emotion = "normal"

        # Verificar emoción (prioridad: excited > curious > scared)
        if any(keyword in segment for keyword in excited_keywords):
            emotion = "excited"
        elif any(keyword in segment for keyword in curious_keywords):
            emotion = "curious"
        elif any(keyword in segment for keyword in scared_keywords):
            emotion = "scared"

        emotions.append(emotion)

    return emotions

# ---------------------------------------------------------------------------
# Schedule de parpadeo con variación Phytom
# ---------------------------------------------------------------------------

def generate_phytom_blink_schedule(duration: float, fps: int,
                                   min_interval: float = 1.5,
                                   max_interval: float = 4.0,
                                   blink_duration: float = 0.2) -> list:
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
# Renderizado del personaje Phytom frame-by-frame
# ---------------------------------------------------------------------------

def load_upper_background(target_w: int = VIDEO_W, target_h: int = HALF_H) -> Image.Image:
    """Carga el fondo con estilo espacial."""
    # Intentar cargar fondo espacial específico
    space_bg_paths = [
        BACKGROUNDS_DIR / "space_background.png",
        BACKGROUNDS_DIR / "space.jpg",
        BACKGROUNDS_DIR / "cosmic.png"
    ]

    for bg_path in space_bg_paths:
        if bg_path.exists():
            bg = Image.open(str(bg_path)).convert("RGB")
            # Resize al ancho del video manteniendo proporcion
            ratio = target_w / bg.width
            new_h = int(bg.height * ratio)
            bg = bg.resize((target_w, new_h), Image.LANCZOS)
            # Recortar la parte superior
            if new_h > target_h:
                bg = bg.crop((0, 0, target_w, target_h))
            elif new_h < target_h:
                # Si es más pequeño, centrar sobre fondo oscuro espacial
                canvas = Image.new("RGB", (target_w, target_h), COLOR_BG_TOP)
                y_off = (target_h - new_h) // 2
                canvas.paste(bg, (0, y_off))
                bg = canvas
            print(f"  [Background] Fondo espacial: {bg_path.name}")
            return bg

    # Usar podcast_studio.png como fallback
    bg_path = BACKGROUNDS_DIR / "podcast_studio.png"
    if bg_path.exists():
        bg = Image.open(str(bg_path)).convert("RGB")
        ratio = target_w / bg.width
        new_h = int(bg.height * ratio)
        bg = bg.resize((target_w, new_h), Image.LANCZOS)
        if new_h > target_h:
            bg = bg.crop((0, 0, target_w, target_h))
        elif new_h < target_h:
            canvas = Image.new("RGB", (target_w, target_h), COLOR_BG_TOP)
            y_off = (target_h - new_h) // 2
            canvas.paste(bg, (0, y_off))
            bg = canvas
        print(f"  [Background] Fallback: podcast_studio.png")
        return bg
    else:
        print("  [Background] Sin fondo. Usando color sólido espacial.")
        return Image.new("RGB", (target_w, target_h), COLOR_BG_TOP)

def build_phytom_clip(sprites: dict, amplitudes: np.ndarray,
                     blink_schedule: list, emotion_schedule: list,
                     duration: float, fps: int,
                     bg_image: Image.Image = None,
                     target_w: int = VIDEO_W, target_h: int = HALF_H):
    """
    Construye un VideoClip de la mitad superior con Phytom animado.
    Incluye efectos de brillo y colores según emoción.
    """
    from moviepy.editor import VideoClip

    # Cargar fondo si no se pasó uno
    if bg_image is None:
        bg_image = load_upper_background(target_w, target_h)

    # Escalar sprites para Phytom
    body = sprites["body"]
    max_char_h = int(target_h * 0.85)
    scale = min(max_char_h / body.height, (target_w * 0.9) / body.width)
    if scale < 1.0 or scale > 2.5:
        scale = min(max(scale, 0.5), 2.5)

    scaled_sprites = {}
    for name, img in sprites.items():
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        scaled_sprites[name] = img.resize((new_w, new_h), Image.LANCZOS)

    print(f"  [Phytom] Escala: {scale:.2f}x -> {scaled_sprites['body'].size}")

    char_w = scaled_sprites["body"].width
    char_h = scaled_sprites["body"].height
    char_x = (target_w - char_w) // 2
    char_y = (target_h - char_h) // 2 + 30

    # Umbral RMS para boca abierta (más sensible para Phytom)
    rms_threshold = 0.12

    total_frames = int(duration * fps)
    n_emotions = len(emotion_schedule)

    def make_frame(t):
        frame_idx = min(int(t * fps), total_frames - 1)

        # Usar fondo de imagen
        frame = bg_image.copy()

        # Determinar estado de Phytom
        is_blinking = blink_schedule[frame_idx] if frame_idx < len(blink_schedule) else False
        rms = amplitudes[frame_idx] if frame_idx < len(amplitudes) else 0.0
        is_speaking = rms > rms_threshold

        # Determinar emoción actual
        if n_emotions > 0:
            segment_idx = min(int(frame_idx / total_frames * n_emotions), n_emotions - 1)
            current_emotion = emotion_schedule[segment_idx]
        else:
            current_emotion = "normal"

        # Seleccionar sprite según emoción y estado
        if is_blinking:
            sprite = scaled_sprites["closed"]
        elif current_emotion == "excited" and is_speaking:
            sprite = scaled_sprites["angry"]
        elif is_speaking:
            sprite = scaled_sprites["open"]
        else:
            sprite = scaled_sprites["body"]

        # Efecto de respiración (mayor amplitud para alienígena)
        breath_offset = int(5 * math.sin(2 * math.pi * t / 3.0))
        paste_y = char_y + breath_offset

        # Pegar sprite
        frame.paste(sprite, (char_x, paste_y), sprite)

        # Efecto de brillo según emoción
        if current_emotion == "excited":
            # Añadir brillo cian/azul alrededor
            glow_frame = np.array(frame)
            glow_mask = np.array(sprite)[:, :, 3] > 0  # Máscara del sprite
            glow_frame[glow_mask] = np.clip(
                glow_frame[glow_mask] + np.array([30, 50, 80]), 0, 255
            )
            frame = Image.fromarray(glow_frame)
        elif current_emotion == "curious":
            # Añadir brillo suave
            glow_frame = np.array(frame)
            glow_mask = np.array(sprite)[:, :, 3] > 0
            glow_frame[glow_mask] = np.clip(
                glow_frame[glow_mask] + np.array([20, 20, 40]), 0, 255
            )
            frame = Image.fromarray(glow_frame)

        return np.array(frame)

    clip = VideoClip(make_frame, duration=duration)
    return clip

# ---------------------------------------------------------------------------
# Subtítulos con estilo alienígena
# ---------------------------------------------------------------------------

def _get_phytom_font(size: int):
    """Intenta cargar fuentes futuristas para Phytom."""
    # Lista de fuentes alternativas en orden preferencia
    font_names = [
        "futura.ttf", "Futura.ttf", "orbitron.ttf", "Orbitron.ttf",
        "arialbd.ttf", "Arial Bold.ttf", "impact.ttf", "Impact.ttf"
    ]

    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except (OSError, IOError):
            pass

    # Buscar en carpeta de Windows
    win_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for font_name in ["arialbd.ttf", "arial.ttf", "impact.ttf"]:
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

def _render_phytom_subtitle_image(text: str, font, max_w: int,
                                color=(100, 255, 255), stroke_color=(0, 0, 0),
                                stroke_width: int = 2) -> Image.Image:
    """Renderiza texto con estilo alienígena usando Pillow."""
    # Calcular tamaño del texto
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

        # Dibujar borde negro delgado
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if dx * dx + dy * dy <= stroke_width * stroke_width:
                    draw.text((x + dx, y + dy), line, font=font,
                              fill=(*stroke_color, 255))

        # Dibujar texto principal con color cian
        draw.text((x, y), line, font=font, fill=(*color, 255))
        y += line_h

    return img

def build_phytom_subtitle_clips(guion: str, duration: float,
                               target_w: int = VIDEO_W, subtitle_y: int = HALF_H - 40):
    """
    Genera subtítulos con estilo alienígena.
    Palabra por palabra o frases cortas, centrados.
    """
    from moviepy.editor import ImageClip

    font = _get_phytom_font(90)

    # Dividir en segmentos más cortos para estilo viral
    words = guion.split()
    group_size = 1  # Una palabra a la vez para más impacto
    segments = []
    for i in range(0, len(words), group_size):
        chunk = " ".join(words[i:i + group_size])
        segments.append(chunk.upper())  # Todo mayúscula para estilo alienígena

    if not segments:
        return None

    segment_duration = duration / len(segments)
    clips = []

    for i, text in enumerate(segments):
        start_time = i * segment_duration

        # Renderizar con estilo Phytom
        img = _render_phytom_subtitle_image(text, font, target_w - 80)
        img_array = np.array(img)

        txt_clip = (ImageClip(img_array, ismask=False, transparent=True)
                    .set_start(start_time)
                    .set_duration(segment_duration)
                    .set_position(("center", subtitle_y)))

        clips.append(txt_clip)

    print(f"  [Subtítulos Phytom] {len(segments)} segmentos, {segment_duration:.2f}s cada uno")
    return clips

# ---------------------------------------------------------------------------
# Branding Phytom
# ---------------------------------------------------------------------------

def _render_phytom_branding_image(text: str, badge: str, color=(100, 255, 255)) -> Image.Image:
    """Renderiza branding con estilo Phytom."""
    try:
        font = ImageFont.truetype("arialbd.ttf", 32)
        badge_font = ImageFont.truetype("arial.ttf", 24)
    except (OSError, IOError):
        try:
            font = ImageFont.load_default(size=32)
            badge_font = ImageFont.load_default(size=24)
        except TypeError:
            font = ImageFont.load_default()
            badge_font = ImageFont.load_default()

    # Calcular tamaño
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    badge_bbox = badge_font.getbbox(badge)
    badge_w = badge_bbox[2] - badge_bbox[0]
    badge_h = badge_bbox[3] - badge_bbox[1]

    total_w = max(text_w, badge_w) + 40
    total_h = text_h + badge_h + 20

    img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Badge
    badge_y = 5
    draw.text((20, badge_y), badge, font=badge_font, fill=(*color, 200))

    # Texto principal
    text_y = badge_y + badge_h + 10
    draw.text((20, text_y), text, font=font, fill=(*color, 255))

    return img

# ---------------------------------------------------------------------------
# Mezcla de audio con efectos espaciales
# ---------------------------------------------------------------------------

def mix_phytom_audio(narration_path: str, duration: float):
    """Mezcla narración con música espacial de fondo."""
    from moviepy.editor import AudioFileClip, CompositeAudioClip

    narration = AudioFileClip(narration_path)

    # Buscar música espacial
    music_files = []
    # Buscar con nombres espaciales primero
    space_patterns = ["space*", "cosmic*", "stellar*", "ambient*", "lofi*"]
    for pattern in space_patterns:
        music_files.extend(MUSIC_DIR.glob(f"{pattern}.mp3"))

    # Si no hay música espacial, usar cualquier música
    if not music_files:
        music_files = list(MUSIC_DIR.glob("*.mp3"))

    if not music_files:
        print("  [Audio] Sin música de fondo espacial.")
        return narration

    track_path = random.choice(music_files)
    print(f"  [Audio] Música: {track_path.name}")

    try:
        music = AudioFileClip(str(track_path))
        # Volumen más bajo para no competir con la voz
        music = music.volumex(0.08)

        if music.duration < duration:
            from moviepy.audio.fx.audio_loop import audio_loop
            music = audio_loop(music, duration=duration)
        else:
            music = music.subclip(0, duration)

        # Fade más suave
        music = music.audio_fadein(3.0).audio_fadeout(5.0)
        return CompositeAudioClip([narration, music])
    except Exception as e:
        print(f"  [Audio] Error con música: {e}. Usando solo narración.")
        return narration

# ---------------------------------------------------------------------------
# Motor principal de composición Phytom
# ---------------------------------------------------------------------------

def build_phytom_video(audio_path: str, duration: float, guion: str,
                     titulo: str, output_path: str) -> None:
    """
    Compone el video final de Phytom:
      - Mitad superior: Phytom animado con efectos emocionales
      - Mitad inferior: fondo espacial
      - Subtítulos con estilo alienígena
      - Audio: narración + música espacial
    """
    from moviepy.editor import (
        CompositeVideoClip, ColorClip, concatenate_videoclips
    )

    print("\n  === COMPOSICIÓN DE VIDEO PHYTOM ===")

    # --- 1. Fondo de video (mitad inferior) ---
    print("\n  [1/5] Preparando fondo espacial...")
    bg_videos = scan_background_videos(VIDEOS_FONDO_DIR)
    selected_bg = random.choice(bg_videos)
    bg_clip = prepare_background_clip(selected_bg, duration)
    bg_clip = bg_clip.set_position((0, HALF_H))

    # --- 2. Cargar Phytom ---
    print("\n  [2/5] Cargando sprites de Phytom...")
    sprites = load_phytom_sprites(CHARACTERS_DIR)

    # --- 3. Análisis de audio y emociones ---
    print("\n  [3/5] Analizando audio para lip-sync...")
    amplitudes = extract_audio_amplitudes(audio_path, VIDEO_FPS)
    blink_schedule = generate_phytom_blink_schedule(duration, VIDEO_FPS)
    emotion_schedule = detect_phytom_emotions(guion)
    print(f"  [Lip Sync] {len(amplitudes)} frames | "
          f"Parpadeos programados | "
          f"Emociones: {set(emotion_schedule)}")

    # --- 4. Clip de Phytom (mitad superior con fondo espacial) ---
    print("\n  [4/5] Renderizando Phytom animado...")
    upper_bg = load_upper_background()
    phytom_clip = build_phytom_clip(
        sprites, amplitudes, blink_schedule, emotion_schedule,
        duration, VIDEO_FPS, bg_image=upper_bg
    )
    phytom_clip = phytom_clip.set_position((0, 0))

    # --- 5. Subtítulos ---
    print("\n  [5/5] Generando subtítulos alienígenas...")
    subtitle_clips = build_phytom_subtitle_clips(guion, duration)

    # --- Composición final ---
    print("\n  [Render] Componiendo video final...")

    # Fondo negro completo como base
    base = ColorClip(size=(VIDEO_W, VIDEO_H), color=COLOR_BG_TOP).set_duration(duration)

    layers = [base, phytom_clip, bg_clip]
    if subtitle_clips:
        layers.extend(subtitle_clips)

    # Branding Phytom
    try:
        from moviepy.editor import ImageClip as _IC
        branding_img = _render_phytom_branding_image(PHYTOM_BRANDING, PHYTOM_BADGE)
        branding_arr = np.array(branding_img)
        branding = (_IC(branding_arr, ismask=False, transparent=True)
                    .set_duration(duration)
                    .set_position(("center", VIDEO_H - 80)))
        layers.append(branding)
    except Exception:
        pass

    video = CompositeVideoClip(layers, size=(VIDEO_W, VIDEO_H))

    # --- Audio ---
    print("  [Audio] Mezclando audio con efectos espaciales...")
    final_audio = mix_phytom_audio(audio_path, duration)
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
    print(f"  [OK] Video de Phytom generado: {output_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phytom Video Generator - Video de Phytom Animado",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo:
  python phytom_video_gen.py \\
    --guion "¡Hola! Soy Phytom, tu amigo alienígena del espacio..." \\
    --titulo "Phytom Te Saluda" \\
    --output ./phytom_video.mp4

Assets requeridos:
  assets/character/          body.png, character_open.png, character_closed.png, character_angry.png
  assets/videos_fondo/        *.mp4 (fondos espaciales o videos existentes)
  assets/music/               *.mp3 (opcional, música espacial)

Variables de entorno:
  AZURE_SPEECH_KEY    -> Clave de Azure Cognitive Services
  AZURE_SPEECH_REGION -> Region (ej: francecentral)
  PHYTOM_VOICE       -> Voz para Phytom (default: es-ES-LauraNeural)
        """
    )

    parser.add_argument("--guion", required=True,
                        help="Texto completo del guión para Phytom.")
    parser.add_argument("--titulo", required=True,
                        help="Título del video de Phytom.")
    parser.add_argument("--output", default="phytom_video.mp4",
                        help="Ruta del MP4 de salida. (default: phytom_video.mp4)")

    return parser.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print(" Phytom Video Generator 👽")
    print("=" * 60)
    print(f"  Título: {args.titulo}")
    print(f"  Guion:  {args.guion[:80]}...")
    print(f"  Voz:    {get_phytom_voice()}")

    tmp_audio = os.path.join(
        tempfile.gettempdir(),
        f"phytom_audio_{uuid.uuid4().hex}.mp3"
    )

    try:
        # 1. Generar audio
        print("\n[1/3] Generando audio con Azure TTS (voz de Phytom)...")
        generate_audio_phytom(text=args.guion, output_path=tmp_audio)

        # 2. Medir duración
        duration = get_audio_duration(tmp_audio)
        print(f"  Duración del audio: {duration:.2f}s")

        # 3. Construir video
        print("\n[2/3] Construyendo video de Phytom...")
        build_phytom_video(
            audio_path=tmp_audio,
            duration=duration,
            guion=args.guion,
            titulo=args.titulo,
            output_path=args.output,
        )

        print(f"\n[3/3] ¡Listo!")
        print(f"  Video: {os.path.abspath(args.output)}")
        print(f"OUTPUT_PATH:{os.path.abspath(args.output)}")
        print("\n  ¡Phytom está listo para conquistar el espacio! 🚀")

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