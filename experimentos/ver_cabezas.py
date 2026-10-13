#!/usr/bin/env python3
"""Pinta, para cada caso, los mapas de unas cabezas concretas y su suma."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from desgrane import _calor, _mosaico

CABEZAS = [(20, 15), (21, 31), (21, 8), (21, 25)]
D = Path("out/desgrane")
casos = json.loads(Path("experimentos/casos.json").read_text())
celdas = []
for caso in casos:
    nombre = f"{Path(caso['imagen']).stem}__{re.sub(r'[^a-z0-9]+', '-', caso['pregunta'].lower()).strip('-')[:36]}"
    z = np.load(D / f"{nombre}.npz")
    a = z["aten"]
    img = Image.open(caso["imagen"]).convert("RGB")
    # el token mas atendido por la media de las capas 24-34: si no cambia con el predicado, es un sumidero
    tardia = a[24:35].mean((0, 1))
    y, x = np.unravel_index(tardia.argmax(), tardia.shape)
    print(f"{nombre[:44]:44s} sumidero tardio en (fila {y}, col {x}) con {tardia.max() / tardia.sum():.0%} de la masa")
    for c, h in CABEZAS:
        celdas.append(_calor(img, a[c, h], f"{caso['palabra']}  c{c} h{h}  masa {a[c, h].sum():.2f}"))
    suma = sum(a[c, h] for c, h in CABEZAS)
    celdas.append(_calor(img, suma, f"{caso['palabra']}  suma de las 4"))
_mosaico(celdas, len(CABEZAS) + 1).save(D / "cabezas.png")
