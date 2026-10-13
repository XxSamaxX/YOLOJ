#!/usr/bin/env python3
"""Cuanto se parece el mapa de atencion, umbralizado, a la mascara real del objeto.

Sin GPU: usa lo que guardo coco_atencion.py. El umbral se elige en la mitad A y
se mide en la B. Todo a resolucion de rejilla (celdas de 32 px), que es el techo
de lo que la atencion puede dar sin otro modelo que afine el borde.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from coco_cabezas import cargar

# las 5 mejores cabezas en la variante relativa, elegidas en A por coco_cabezas.py
CABEZAS = [(20, 15), (20, 4), (21, 18), (21, 31), (22, 6)]
UMBRALES = np.linspace(0.05, 0.9, 18)


def mapa(z) -> np.ndarray:
    pos = z["pos"]["aten"].astype(np.float32)
    gen = z["gen"]["aten"].astype(np.float32)
    rel = [pos[c, h] / (gen[c, h] + 1e-4) for c, h in CABEZAS]
    m = sum(r / (r.sum() + 1e-9) for r in rel)
    # media 3x3: un objeto ocupa celdas vecinas, un pico suelto no
    p = np.pad(m, 1, mode="edge")
    m = sum(p[i:i + m.shape[0], j:j + m.shape[1]] for i in range(3) for j in range(3)) / 9
    return m / m.max()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def caja(m: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(m)
    c = np.zeros_like(m)
    if len(ys):
        c[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return c


def recoger(d: Path, mitad: str):
    return [(f, mapa(z), z["pos"]["mascara"] >= 0.5) for f, z, _ in cargar(d, mitad)
            if (z["pos"]["mascara"] >= 0.5).any()]


if __name__ == "__main__":
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "out/coco")
    A, B = recoger(d, "A"), recoger(d, "B")
    medias = [np.mean([iou(m >= t, obj) for _, m, obj in A]) for t in UMBRALES]
    t = float(UMBRALES[int(np.argmax(medias))])
    print(f"umbral elegido en A: {t:.2f} del maximo (IoU medio en A {max(medias):.3f})\n")

    filas = []
    for f, m, obj in B:
        pred = m >= t
        oraculo = max(iou(m >= u, obj) for u in UMBRALES)
        filas.append((iou(pred, obj), iou(caja(pred), caja(obj)), oraculo, f["instancias"], f["area"]))
    v = np.array(filas)
    print(f"mitad B, {len(v)} imagenes, a resolucion de rejilla:")
    print(f"   IoU de mascara           {v[:, 0].mean():.3f}   (con el mejor umbral por imagen: {v[:, 2].mean():.3f})")
    print(f"   IoU de caja              {v[:, 1].mean():.3f}")
    print(f"   cajas con IoU >= 0,5     {(v[:, 1] >= 0.5).mean():.1%}")
    for nombre, filtro in (("una sola instancia", v[:, 3] == 1), ("varias instancias", v[:, 3] > 1),
                           ("objeto < 3 % de la imagen", v[:, 4] < 0.03), ("objeto >= 3 %", v[:, 4] >= 0.03)):
        s = v[filtro]
        print(f"   {nombre:26s} n={len(s):3d}  IoU mascara {s[:, 0].mean():.3f}  caja {s[:, 1].mean():.3f}"
              f"  cajas >= 0,5: {(s[:, 1] >= 0.5).mean():.0%}")
