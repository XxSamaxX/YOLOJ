#!/usr/bin/env python3
"""Funciona tambien sobre capturas de pantalla? Elementos de interfaz con sitio conocido.

    pantalla.py out/pantalla/casos.json [--lado 960 1600]

casos.json: {"captura.png": [[pregunta, [[x0, y0, x1, y1], ...] | null], ...]}
Con cajas, se espera un "si" y que el punto caiga dentro de alguna; con null, un "no".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from coco_atencion import pregunta
from desgrane import GENERICO, cargar
from directo import COLORES, pico
from foto import mapa_relativo
from prefijo import Imagen


def distancia(x: float, y: float, cajas) -> float:
    """Pixeles del punto a la caja mas cercana; 0 si cae dentro."""
    return min(float(np.hypot(max(c[0] - x, 0, x - c[2]), max(c[1] - y, 0, y - c[3]))) for c in cajas)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("casos")
    p.add_argument("--lado", type=int, nargs="+", default=[960, 1600])
    p.add_argument("--lote", type=int, default=3)
    a = p.parse_args()
    carpeta = Path(a.casos).parent
    casos = json.loads(Path(a.casos).read_text())
    modelo, proc = cargar(atenciones=True)
    fuente = ImageFont.load_default(size=22)

    for lado in a.lado:
        n_si = ok_si = dentro = cerca = n_no = ok_no = 0
        for nombre, lista in casos.items():
            img = Image.open(carpeta / nombre).convert("RGB")
            f = min(1.0, lado / max(img.size))
            pre = Imagen(modelo, proc, img.resize((round(img.width * f), round(img.height * f)), Image.LANCZOS))
            textos = [q if q.endswith("?") else pregunta(q) for q, _ in lista] + [GENERICO]
            gaps, atens = [], []
            for i in range(0, len(textos), a.lote):
                g, at = pre.preguntar(textos[i:i + a.lote])
                gaps.extend(g)
                atens.extend(at)
                torch.cuda.empty_cache()
            print(f"\n{nombre}, lado {lado}: rejilla {pre.rejilla[0]}x{pre.rejilla[1]}, una celda = {img.width / pre.rejilla[1]:.0f} px de pantalla")
            lienzo = img.copy()
            d = ImageDraw.Draw(lienzo)
            for k, ((q, cajas), gap, at) in enumerate(zip(lista, gaps, atens)):
                if cajas is None:
                    n_no += 1
                    ok_no += gap <= 0
                    print(f"   {q:48s} {gap:+6.1f}   {'bien, no esta' if gap <= 0 else 'MAL: dice que si'}")
                    continue
                n_si += 1
                ok_si += gap > 0
                x, y = pico(mapa_relativo(at, atens[-1]))
                x, y = x * img.width, y * img.height
                dist = distancia(x, y, cajas)
                dentro += dist == 0
                cerca += dist <= 40
                print(f"   {q:48s} {gap:+6.1f}   {'DENTRO' if dist == 0 else f'a {dist:4.0f} px'}{'' if gap > 0 else '   (dice que no)'}")
                color = COLORES[k % len(COLORES)]
                for c in cajas:
                    d.rectangle(c, outline=color, width=2)
                d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=color, outline=(255, 255, 255), width=3)
                d.text((x + 14, y), q, font=fuente, anchor="lm", fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
            lienzo.save(carpeta / f"{Path(nombre).stem}__{lado}.jpg", quality=88)
        print(f"\n== lado {lado}: presentes dados por presentes {ok_si}/{n_si}; punto dentro del elemento {dentro}/{n_si}, "
              f"a 40 px o menos {cerca}/{n_si}; ausentes dados por ausentes {ok_no}/{n_no}"
              f"   (pico de VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)")


if __name__ == "__main__":
    main()
