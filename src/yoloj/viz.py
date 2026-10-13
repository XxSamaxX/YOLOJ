"""Dibujo de la piramide: un panel por nivel con el gap de cada celda."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .piramide import Resultado

VERDE, ROJO, GRIS = (40, 200, 90), (225, 60, 60), (120, 120, 120)


def _fuente(px: int):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:               # Pillow < 10.1 no admite tamano
        return ImageFont.load_default()


def texto(resultado: Resultado) -> str:
    """El mapa en texto: un bloque por nivel, '·' en las celdas podadas."""
    lineas = []
    for i, n in enumerate(resultado.niveles):
        lineas.append(f"nivel {i} ({n}x{n})")
        for fila in resultado.nivel(i):
            lineas.append("  " + " ".join(
                f"{c.gap:+6.1f}" if c.estado == "evaluada" else "     ·" for c in fila))
    return "\n".join(lineas)


def panel(imagen: Image.Image | str, resultado: Resultado, escala: float = 10.0,
          ancho_panel: int = 480) -> Image.Image:
    """Un panel por nivel, lado a lado.

    Verde = si, rojo = no; la opacidad crece con |gap| hasta `escala` nats.
    Las celdas podadas van en gris: no se evaluaron, heredan el "no" de arriba.
    """
    if isinstance(imagen, str):
        imagen = Image.open(imagen)
    base = imagen.convert("RGB")
    f = ancho_panel / base.width
    base = base.resize((ancho_panel, round(base.height * f)), Image.BICUBIC)
    w, h = base.size
    cabecera, margen = 26, 8
    k = len(resultado.niveles)
    lienzo = Image.new("RGB", (k * w + (k + 1) * margen, h + cabecera + margen), (18, 18, 18))
    rotulo = _fuente(15)

    for i, n in enumerate(resultado.niveles):
        capa = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(capa)
        letra = _fuente(max(11, min(22, w // (n * 5))))
        for fila in resultado.nivel(i):
            for c in fila:
                # la rejilla nominal, no la caja con solape: los paneles no se pisan
                x0, y0 = c.col * w / n, c.fila * h / n
                x1, y1 = (c.col + 1) * w / n, (c.fila + 1) * h / n
                if c.estado == "podada":
                    relleno, marca = GRIS + (150,), "·"
                else:
                    color = VERDE if c.gap > 0 else ROJO
                    alfa = int(40 + 140 * min(abs(c.gap) / escala, 1.0))
                    relleno, marca = color + (alfa,), f"{c.gap:+.1f}"
                d.rectangle([x0, y0, x1, y1], fill=relleno, outline=(255, 255, 255, 200))
                d.text(((x0 + x1) / 2, (y0 + y1) / 2), marca, font=letra, anchor="mm",
                       fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        x = margen + i * (w + margen)
        lienzo.paste(Image.alpha_composite(base.convert("RGBA"), capa).convert("RGB"), (x, cabecera))
        evaluadas = sum(c.estado == "evaluada" for fila in resultado.nivel(i) for c in fila)
        ImageDraw.Draw(lienzo).text((x, 5), f"{n}x{n}  ·  {evaluadas}/{n * n} evaluadas",
                                    font=rotulo, fill=(235, 235, 235))
    return lienzo
