#!/usr/bin/env python3
"""
carousel_gen.py — Tu Dosis Financiera (WF06)
=============================================
Reemplaza Creatomate para la Fábrica de Carruseles.

Genera 5 imágenes JPG 1080x1080 con estética "Dark Finance Minimal":
  Slide 1: Portada (título + subtítulo)
  Slides 2-4: Valor (título + cuerpo)
  Slide 5: CTA (llamada a la acción + suscripción)

Uso:
  python carousel_gen.py \\
    --slides_json '{"slide1":{"titulo":"EL AHORRO QUE CAMBIA TODO","subtitulo":"Lo que nadie te enseñó"},...}' \\
    --categoria "AHORRO" \\
    --output_dir ./carruseles \\
    --fecha 2026-02-17

Salida:
  slide_1_2026-02-17.jpg
  slide_2_2026-02-17.jpg
  ...
  slide_5_2026-02-17.jpg

Imprime las rutas absolutas al stdout (una por línea) para que n8n las capture.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constantes de diseño — Dark Finance Minimal
# ---------------------------------------------------------------------------
CANVAS_W = 1080
CANVAS_H = 1080

# Paleta de colores (R, G, B)
C_BG = (10, 10, 10)           # Fondo negro casi puro
C_WHITE = (255, 255, 255)      # Texto principal
C_GOLD = (212, 175, 55)        # Acento dorado
C_GRAY = (170, 170, 170)       # Subtexto gris
C_BADGE_BG = (26, 26, 26)      # Fondo de badges
C_LINE = (212, 175, 55)        # Color de líneas separadoras

BRANDING = "TuDosisFinanciera.com"
HANDLE = "@TuDosisFinanciera"

# Rutas de fuentes
SCRIPT_DIR = Path(__file__).parent
FONTS_DIR = SCRIPT_DIR.parent / "assets" / "fonts"
FONT_BOLD = str(FONTS_DIR / "Montserrat-Bold.ttf")
FONT_REGULAR = str(FONTS_DIR / "Inter-Regular.ttf")
FONT_SEMIBOLD = str(FONTS_DIR / "Inter-SemiBold.ttf")


# ---------------------------------------------------------------------------
# Utilidades de fuentes y texto
# ---------------------------------------------------------------------------

def load_font(font_path: str, size: int):
    """Carga fuente TTF con fallback a fuente por defecto de Pillow."""
    from PIL import ImageFont
    try:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)
    except Exception:
        pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_wrapped_text(draw, text: str, font, x: int, y: int,
                       max_width: int, color: tuple,
                       line_spacing: int = 10, align: str = "center") -> int:
    """
    Dibuja texto con ajuste de línea automático.
    Retorna la coordenada Y final (bottom del texto dibujado).
    """
    words = text.split()
    lines = []
    current = []

    for word in words:
        test = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))

    cur_y = y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        line_h = bbox[3] - bbox[1]

        if align == "center":
            line_x = x + (max_width - line_w) // 2
        elif align == "right":
            line_x = x + max_width - line_w
        else:
            line_x = x

        draw.text((line_x, cur_y), line, font=font, fill=color)
        cur_y += line_h + line_spacing

    return cur_y


def measure_wrapped_height(draw, text: str, font, max_width: int,
                            line_spacing: int = 10) -> int:
    """Calcula la altura total que ocupará el texto con ajuste de línea."""
    words = text.split()
    lines = []
    current = []

    for word in words:
        test = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))

    if not lines:
        return 0

    sample_h = draw.textbbox((0, 0), "Ag", font=font)[3]
    return sample_h * len(lines) + line_spacing * (len(lines) - 1)


# ---------------------------------------------------------------------------
# Generadores de slides
# ---------------------------------------------------------------------------

def draw_common_elements(draw, categoria: str, slide_num: int,
                          total: int = 5, show_counter: bool = True):
    """Dibuja elementos comunes a todos los slides: badge, footer, contador."""
    from PIL import ImageDraw

    # --- Badge categoría (esquina superior izquierda) ---
    if categoria:
        badge_font = load_font(FONT_SEMIBOLD, 28)
        badge_text = f"  {categoria.upper()}  "
        bbox = draw.textbbox((0, 0), badge_text, font=badge_font)
        bw = bbox[2] - bbox[0] + 24
        bh = bbox[3] - bbox[1] + 16
        bx, by = 40, 40
        draw.rounded_rectangle([(bx, by), (bx + bw, by + bh)],
                                radius=10, fill=C_BADGE_BG)
        draw.text((bx + 12, by + 8), badge_text, font=badge_font, fill=C_GOLD)

    # --- Contador de slide (esquina superior derecha) ---
    if show_counter:
        counter_font = load_font(FONT_SEMIBOLD, 28)
        counter_text = f"{slide_num}/{total}"
        bbox = draw.textbbox((0, 0), counter_text, font=counter_font)
        cw = bbox[2] - bbox[0]
        draw.text((CANVAS_W - cw - 50, 52), counter_text,
                  font=counter_font, fill=C_GOLD)

    # --- Footer de branding (parte inferior centrado) ---
    footer_font = load_font(FONT_REGULAR, 24)
    fbbox = draw.textbbox((0, 0), BRANDING, font=footer_font)
    fw = fbbox[2] - fbbox[0]
    draw.text(((CANVAS_W - fw) // 2, CANVAS_H - 50),
              BRANDING, font=footer_font, fill=C_GOLD)


def draw_gold_line(draw, x: int, y: int, width: int = 120, height: int = 3):
    """Dibuja la línea dorada decorativa característica del diseño."""
    draw.rectangle([(x, y), (x + width, y + height)], fill=C_LINE)


def generate_slide_1(slide_data: dict, categoria: str) -> object:
    """
    Slide 1 — Portada
    Layout: badge + línea dorada + TÍTULO en mayúsculas + subtítulo + footer
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), color=C_BG)
    draw = ImageDraw.Draw(img)

    titulo = slide_data.get("titulo", "").upper()
    subtitulo = slide_data.get("subtitulo", "")

    margin = 80
    max_w = CANVAS_W - margin * 2

    font_titulo = load_font(FONT_BOLD, 80)
    font_subtitulo = load_font(FONT_REGULAR, 40)

    # Calcular alturas para centrar el bloque verticalmente
    titulo_h = measure_wrapped_height(draw, titulo, font_titulo, max_w, line_spacing=12)
    subtitulo_h = measure_wrapped_height(draw, subtitulo, font_subtitulo, max_w, line_spacing=8)

    line_h = 4
    gap = 40  # espacio entre elementos
    block_h = line_h + gap + titulo_h + gap + line_h + gap + subtitulo_h
    start_y = (CANVAS_H - block_h) // 2

    # Línea superior
    line_w = 100
    draw_gold_line(draw, (CANVAS_W - line_w) // 2, start_y, line_w, line_h)
    cur_y = start_y + line_h + gap

    # Título
    cur_y = draw_wrapped_text(draw, titulo, font_titulo,
                              margin, cur_y, max_w, C_WHITE,
                              line_spacing=12, align="center")
    cur_y += gap

    # Línea inferior
    draw_gold_line(draw, (CANVAS_W - line_w) // 2, cur_y, line_w, line_h)
    cur_y += line_h + gap

    # Subtítulo
    draw_wrapped_text(draw, subtitulo, font_subtitulo,
                      margin, cur_y, max_w, C_GRAY,
                      line_spacing=8, align="center")

    # Elementos comunes (sin contador en portada)
    draw_common_elements(draw, categoria, slide_num=1, show_counter=False)

    return img


def generate_slide_valor(slide_data: dict, categoria: str, slide_num: int) -> object:
    """
    Slides 2-4 — Valor
    Layout: badge + título + línea dorada + cuerpo + contador + footer
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), color=C_BG)
    draw = ImageDraw.Draw(img)

    titulo = slide_data.get("titulo", "")
    cuerpo = slide_data.get("cuerpo", "")

    margin = 80
    max_w = CANVAS_W - margin * 2

    font_titulo = load_font(FONT_BOLD, 64)
    font_cuerpo = load_font(FONT_REGULAR, 42)

    titulo_h = measure_wrapped_height(draw, titulo, font_titulo, max_w, line_spacing=10)
    cuerpo_h = measure_wrapped_height(draw, cuerpo, font_cuerpo, max_w, line_spacing=10)

    line_h = 3
    gap = 36
    block_h = titulo_h + gap + line_h + gap + cuerpo_h
    start_y = (CANVAS_H - block_h) // 2

    # Título
    cur_y = draw_wrapped_text(draw, titulo, font_titulo,
                              margin, start_y, max_w, C_WHITE,
                              line_spacing=10, align="center")
    cur_y += gap

    # Línea dorada (más corta, debajo del título)
    line_w = 80
    draw_gold_line(draw, (CANVAS_W - line_w) // 2, cur_y, line_w, line_h)
    cur_y += line_h + gap

    # Cuerpo
    draw_wrapped_text(draw, cuerpo, font_cuerpo,
                      margin, cur_y, max_w, C_GRAY,
                      line_spacing=10, align="center")

    # Elementos comunes
    draw_common_elements(draw, categoria, slide_num=slide_num, show_counter=True)

    return img


def generate_slide_5(slide_data: dict, categoria: str) -> object:
    """
    Slide 5 — CTA
    Layout: línea dorada + CTA en dorado grande + subcta en blanco + handle + footer
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), color=C_BG)
    draw = ImageDraw.Draw(img)

    cta = slide_data.get("cta", "").upper()
    subcta = slide_data.get("subcta", "Suscríbete gratis → link en bio")

    margin = 80
    max_w = CANVAS_W - margin * 2

    font_cta = load_font(FONT_BOLD, 72)
    font_subcta = load_font(FONT_REGULAR, 38)
    font_handle = load_font(FONT_SEMIBOLD, 32)

    cta_h = measure_wrapped_height(draw, cta, font_cta, max_w, line_spacing=12)
    subcta_h = measure_wrapped_height(draw, subcta, font_subcta, max_w, line_spacing=8)
    handle_h = 40

    line_h = 4
    gap = 44
    block_h = line_h + gap + cta_h + gap + line_h + gap + subcta_h + gap + handle_h
    start_y = (CANVAS_H - block_h) // 2

    # Línea superior
    line_w = 120
    draw_gold_line(draw, (CANVAS_W - line_w) // 2, start_y, line_w, line_h)
    cur_y = start_y + line_h + gap

    # CTA en dorado
    cur_y = draw_wrapped_text(draw, cta, font_cta,
                              margin, cur_y, max_w, C_GOLD,
                              line_spacing=12, align="center")
    cur_y += gap

    # Línea inferior
    draw_gold_line(draw, (CANVAS_W - line_w) // 2, cur_y, line_w, line_h)
    cur_y += line_h + gap

    # Subcta en blanco
    cur_y = draw_wrapped_text(draw, subcta, font_subcta,
                              margin, cur_y, max_w, C_WHITE,
                              line_spacing=8, align="center")
    cur_y += gap

    # Handle de redes sociales
    handle_bbox = draw.textbbox((0, 0), HANDLE, font=font_handle)
    hw = handle_bbox[2] - handle_bbox[0]
    draw.text(((CANVAS_W - hw) // 2, cur_y), HANDLE, font=font_handle, fill=C_GOLD)

    # Elementos comunes (sin contador en CTA slide)
    draw_common_elements(draw, categoria, slide_num=5, show_counter=False)

    return img


# ---------------------------------------------------------------------------
# Generador principal
# ---------------------------------------------------------------------------

def generate_carousel(slides_data: dict, categoria: str,
                      output_dir: str, fecha: str) -> list[str]:
    """
    Genera los 5 slides y los guarda como JPG.
    Retorna lista de rutas absolutas.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = []

    generators = {
        1: lambda d: generate_slide_1(d, categoria),
        2: lambda d: generate_slide_valor(d, categoria, 2),
        3: lambda d: generate_slide_valor(d, categoria, 3),
        4: lambda d: generate_slide_valor(d, categoria, 4),
        5: lambda d: generate_slide_5(d, categoria),
    }

    for num in range(1, 6):
        slide_key = f"slide{num}"
        slide_data = slides_data.get(slide_key, {})

        print(f"  [Slide {num}/5] Generando...")
        img = generators[num](slide_data)

        filename = f"slide_{num}_{fecha}.jpg"
        filepath = out / filename
        img.save(str(filepath), "JPEG", quality=95, optimize=True)
        print(f"  [Slide {num}/5] Guardado: {filepath}")
        paths.append(str(filepath.resolve()))

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tu Dosis Financiera — Generador de Carruseles (reemplaza Creatomate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo:
  python carousel_gen.py \\
    --slides_json '{"slide1":{"titulo":"EL AHORRO QUE CAMBIA TODO","subtitulo":"Lo que nadie te enseñó"},"slide2":{"titulo":"El primer paso","cuerpo":"Ahorra el 10 por ciento de cada ingreso antes de gastar nada"},"slide3":{"titulo":"El error más común","cuerpo":"Esperar a fin de mes para ahorrar lo que sobra. Nunca sobra nada."},"slide4":{"titulo":"La regla de oro","cuerpo":"Automatiza tu ahorro. Configura una transferencia automática el día que cobras."},"slide5":{"cta":"Empieza hoy mismo","subcta":"Suscríbete gratis en el link de bio"}}' \\
    --categoria "AHORRO" \\
    --output_dir ./test_carrusel \\
    --fecha 2026-02-17
        """
    )

    parser.add_argument("--slides_json", required=True,
                        help="JSON string con los 5 slides (estructura de OpenAI).")
    parser.add_argument("--categoria", default="FINANZAS",
                        help="Categoría del tip/libro.")
    parser.add_argument("--output_dir", default="./carruseles",
                        help="Carpeta de salida para los JPG.")
    parser.add_argument("--fecha", default=None,
                        help="Fecha YYYY-MM-DD para nombrar archivos. Si no se pasa, usa hoy.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 55)
    print(" Tu Dosis Financiera — Generador de Carruseles")
    print("=" * 55)

    # Parsear JSON de slides
    try:
        slides_data = json.loads(args.slides_json)
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON de slides inválido: {e}", file=sys.stderr)
        sys.exit(1)

    # Fecha
    if args.fecha:
        fecha = args.fecha
    else:
        from datetime import date
        fecha = date.today().isoformat()

    print(f"  Categoría:  {args.categoria}")
    print(f"  Fecha:      {fecha}")
    print(f"  Salida:     {args.output_dir}")
    print(f"  Slides:     {list(slides_data.keys())}")

    print("\n[Generando carrusel...]")
    paths = generate_carousel(
        slides_data=slides_data,
        categoria=args.categoria,
        output_dir=args.output_dir,
        fecha=fecha,
    )

    print(f"\n[OK] {len(paths)} slides generados:")
    for p in paths:
        print(f"  {p}")

    # Imprimir rutas en formato que n8n puede parsear del stdout
    print("\n--- OUTPUT_PATHS ---")
    for i, p in enumerate(paths, 1):
        print(f"SLIDE_{i}:{p}")
    print("--- END_OUTPUT_PATHS ---")

    print("=" * 55)


if __name__ == "__main__":
    main()
