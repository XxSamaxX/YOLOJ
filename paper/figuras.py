#!/usr/bin/env python3
"""Figuras del paper.

    figuras.py barras                 las de barras, con las cifras de las tablas
    figuras.py ejemplos               atencion / atencion + punto / SAM 3 por texto sobre imagenes de COCO
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AQUI = Path(__file__).parent
FIG = AQUI / "figures"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
GRIS, AZUL, ROJO = "#9a9a9a", "#2b6cb0", "#c53030"


def barras() -> None:
    # pointing game, mitad B (coco_cabezas.py)
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    nombres = ["chance", "image centre", "all 1152 heads, raw", "head 20.15, raw", "head 20.15, relative", "5 heads, relative"]
    v = [10.6, 29.0, 1.5, 59.5, 71.5, 78.5]
    ax.barh(range(len(v)), v, color=[GRIS, GRIS, GRIS, AZUL, AZUL, ROJO], height=0.7)
    ax.set_yticks(range(len(v)), nombres)
    ax.invert_yaxis()
    for i, x in enumerate(v):
        ax.text(x + 1.2, i, f"{x:.1f}", va="center")
    ax.set_xlim(0, 92)
    ax.set_xlabel("pointing-game hit rate (%), 200 held-out images")
    fig.savefig(FIG / "pointing.pdf", bbox_inches="tight")

    # cajas con IoU >= 0,5, una instancia / varias (coco_mascara, coco_sam, coco_sam3texto)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    nombres = ["attention\nalone", "+ SAM 2.1\nsmall", "+ SAM 2.1\nlarge", "+ SAM 3\n(point)", "SAM 3\n(text)"]
    todas, una, varias = [42.0, 54.5, 58.0, 59.5, 85.0], [39, 69, 74, 73, 87], [48, 24, 26, 32, 82]
    x = np.arange(len(nombres))
    ax.bar(x - 0.27, todas, 0.26, color="#333333", label="all")
    ax.bar(x, una, 0.26, color=AZUL, label="one instance")
    ax.bar(x + 0.27, varias, 0.26, color=ROJO, label="several instances")
    ax.set_xticks(x, nombres)
    ax.set_ylabel("boxes with IoU $\\geq$ 0.5 (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=7, columnspacing=1.0, handlelength=1.2)
    fig.savefig(FIG / "boxes.pdf", bbox_inches="tight")


def ejemplos() -> None:
    import torch
    from PIL import Image

    sys.path.insert(0, str(AQUI.parent / "experimentos"))
    from coco_atencion import COCO, pregunta
    from desgrane import GENERICO, cargar
    from foto import Sam3Texto, cargar_sam, mapa_relativo, mascara_sam
    from prefijo import Imagen

    casos = [tuple(c.split(":")) for c in sys.argv[2:]] or [("000000504580.jpg", "giraffe"), ("000000561465.jpg", "donut")]
    # por fases: los tres modelos a la vez no caben en 12 GB
    modelo, proc = cargar(atenciones=True)
    previos = []
    for fichero, clase in casos:
        img = Image.open(COCO / "val2017" / fichero).convert("RGB")
        gaps, at = Imagen(modelo, proc, img).preguntar([pregunta(clase), GENERICO])
        previos.append((img, float(gaps[0]), mapa_relativo(at[0], at[1])))
    del modelo
    torch.cuda.empty_cache()
    sam, proc_sam = cargar_sam("facebook/sam2.1-hiera-large")
    sam3 = Sam3Texto()
    colores = [(0.9, 0.2, 0.2), (0.2, 0.75, 0.3), (0.2, 0.45, 1.0), (1.0, 0.75, 0.1), (0.85, 0.3, 0.85), (0.2, 0.8, 0.8)]

    def pintar(ax, img, mascaras, titulo):
        lienzo = np.asarray(img).astype(float) / 255
        for k, m in enumerate(mascaras):
            c = np.array(colores[k % len(colores)])
            lienzo[m] = 0.45 * lienzo[m] + 0.55 * c
        ax.imshow(lienzo)
        ax.set_title(titulo, fontsize=6.5)
        ax.axis("off")

    fig, ejes = plt.subplots(len(casos), 3, figsize=(7.0, 1.95 * len(casos)))
    for fila, (fichero, clase), (img, gap, m) in zip(np.atleast_2d(ejes), casos, previos):
        masc, punto = mascara_sam(sam, proc_sam, img, m)
        instancias = sam3.imagen(img) or sam3.buscar(clase, 0.5)
        calor = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(img.size, Image.BICUBIC)) / 255
        fila[0].imshow(img.convert("L"), cmap="gray")
        fila[0].imshow(np.dstack([np.ones_like(calor), 0.1 * np.ones_like(calor), 0.1 * np.ones_like(calor), 0.8 * calor]))
        fila[0].plot(*punto, "o", ms=5, mfc="white", mec="black")
        fila[0].set_title(f'"{clase}" (gap {gap:+.1f}): relative attention', fontsize=6.5)
        fila[0].axis("off")
        pintar(fila[1], img, [masc], "attention peak + SAM 2.1 (1 mask)")
        fila[1].plot(*punto, "o", ms=5, mfc="white", mec="black")
        pintar(fila[2], img, [x for x, _ in instancias], f"SAM 3 by text ({len(instancias)} instances)")
        print(fichero, clase, f"gap {gap:+.1f}", len(instancias), "instancias")
    fig.subplots_adjust(wspace=0.03, hspace=0.16)
    fig.savefig(FIG / "examples.pdf", bbox_inches="tight", dpi=170)
    fig.savefig(FIG / "examples.png", bbox_inches="tight", dpi=110)


if __name__ == "__main__":
    {"barras": barras, "ejemplos": ejemplos}[sys.argv[1]]()
