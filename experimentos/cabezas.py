#!/usr/bin/env python3
"""Que cabezas de atencion senalan el objeto del predicado.

Para cada cabeza: fraccion de su atencion a la imagen que cae dentro de la caja
del objeto, frente a lo que tocaria por area. Las cajas son a ojo, en fraccion de
imagen; valen para ordenar cabezas, no para dar una cifra final.
"""
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "out/desgrane")
# (x0, y0, x1, y1) en fraccion de ancho y alto
CAJAS = {
    "gatos_sofa__is-there-a-remote-control-in-the-ima": [(0.08, 0.16, 0.29, 0.25), (0.53, 0.16, 0.60, 0.39)],
    "sala_tv__is-there-a-television-in-the-image": [(0.02, 0.38, 0.23, 0.61)],
}


def mascara(cajas, gh, gw):
    m = np.zeros((gh, gw), bool)
    for x0, y0, x1, y1 in cajas:
        m[int(y0 * gh):int(np.ceil(y1 * gh)), int(x0 * gw):int(np.ceil(x1 * gw))] = True
    return m


punt = {}
for nombre, cajas in CAJAS.items():
    a = np.load(D / f"{nombre}.npz")["aten"]              # [capas, cabezas, gh, gw]
    m = mascara(cajas, *a.shape[2:])
    total = a.sum((2, 3))
    dentro = (a * m).sum((2, 3)) / (total + 1e-9)
    punt[nombre] = (dentro, total, m.mean())
    print(f"\n{nombre}: la caja ocupa {m.mean():.1%} de la rejilla")
    print(f"  media de todas las cabezas: {dentro.mean():.1%} dentro")
    orden = np.dstack(np.unravel_index(np.argsort(-dentro, axis=None), dentro.shape))[0]
    print("  mejores cabezas con atencion a la imagen > 0.05:")
    n = 0
    for c, h in orden:
        if total[c, h] > 0.05:
            print(f"    capa {c:2d} cabeza {h:2d}: {dentro[c, h]:.0%} dentro, {total[c, h]:.2f} de su atencion va a la imagen")
            n += 1
            if n == 12:
                break

(d1, t1, _), (d2, t2, _) = punt.values()
ambas = np.minimum(d1, d2) * ((t1 > 0.05) & (t2 > 0.05))
print("\ncabezas que senalan en LAS DOS imagenes (minimo de ambas):")
for i in np.argsort(-ambas, axis=None)[:12]:
    c, h = np.unravel_index(i, ambas.shape)
    print(f"    capa {c:2d} cabeza {h:2d}: mando {d1[c, h]:.0%}, tele {d2[c, h]:.0%}")
print("\npor capa, media de 'dentro' sobre cabezas con atencion a la imagen > 0.05 (mando / tele):")
for c in range(d1.shape[0]):
    f1, f2 = t1[c] > 0.05, t2[c] > 0.05
    print(f"    capa {c:2d}: {d1[c][f1].mean() if f1.any() else 0:5.0%} ({f1.sum():2d} cab.) / {d2[c][f2].mean() if f2.any() else 0:5.0%} ({f2.sum():2d} cab.)")
