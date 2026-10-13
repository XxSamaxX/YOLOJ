#!/usr/bin/env python3
"""Limpiar la mascara de SAM: quedarse con el trozo bueno y tirar los flecos.

Una de cada cuatro mascaras sale en mas de 5 trozos sueltos. Aqui se mide, en la
mitad B y al pixel, si quitar los trozos que sobran mejora la mascara y la caja:

    trozo del punto    el que contiene el pico de atencion (si cae fuera, el mayor)
    trozo mayor        el de mas area
    trozos >= 10 %     todos los que lleguen al 10 % del mayor (objetos partidos por algo delante)

y si conviene limpiar antes o despues de elegir entre las 3 mascaras de SAM.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from coco_atencion import COCO, mascara
from coco_mascara import mapa
from coco_sam import a_pixeles, acuerdo, caja_de, cargar_sam, iou, picos, segmentar


def trozos(masc: np.ndarray) -> torch.Tensor:
    """Etiqueta las componentes conexas a resolucion completa: 0 fuera, un numero por trozo.

    Cada pixel empieza con su numero y en cada paso toma el mayor de sus vecinos
    dentro de la mascara; al estabilizarse queda un numero por trozo.
    """
    t = torch.from_numpy(masc).to("cuda:0")[None, None].float()
    ids = (torch.arange(t.numel(), device=t.device).view_as(t) + 1).float() * t
    while True:
        nuevo = ids
        for _ in range(16):                   # comprobar la parada cuesta mas que el paso
            nuevo = torch.nn.functional.max_pool2d(nuevo, 3, 1, 1) * t
        if torch.equal(nuevo, ids):
            return ids[0, 0]
        ids = nuevo


def limpiar(masc: np.ndarray, punto, modo: str) -> np.ndarray:
    """punto: (x, y) en pixeles. modo: 'punto', 'mayor' o 'grandes'."""
    ids = trozos(masc)
    nums, areas = torch.unique(ids[ids > 0], return_counts=True)
    if len(nums) <= 1:
        return masc
    mayor = nums[areas.argmax()]
    if modo == "grandes":
        return torch.isin(ids, nums[areas >= 0.1 * areas.max()]).cpu().numpy()
    if modo == "punto":
        x = min(int(punto[0]), masc.shape[1] - 1)
        y = min(int(punto[1]), masc.shape[0] - 1)
        if ids[y, x] > 0:
            mayor = ids[y, x]
    return (ids == mayor).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sam", default="facebook/sam2.1-hiera-small")
    p.add_argument("--datos", default="out/coco")
    a = p.parse_args()
    datos = Path(a.datos)
    sam, proc = cargar_sam(a.sam)

    inst = json.loads((COCO / "instances_val2017.json").read_text())
    ids = {c["name"]: c["id"] for c in inst["categories"]}
    anots = defaultdict(list)
    for an in inst["annotations"]:
        anots[(an["image_id"], an["category_id"])].append(an)

    filas = defaultdict(list)
    ms, fuera, n_tr = [], [], defaultdict(list)
    for f in json.loads((datos / "indice.json").read_text()):
        if f["mitad"] != "B":
            continue
        z = {t: np.load(datos / f"{f['id']:012d}_{t}.npz") for t in ("pos", "gen")}
        m = mapa(z)
        img = Image.open(COCO / "val2017" / f["imagen"]).convert("RGB")
        w, h = img.size
        real = np.asarray(mascara(anots[(f["id"], ids[f["positiva"]])], w, h)) > 0
        if not real.any():
            continue
        punto = a_pixeles(picos(m, 1), m.shape, w, h)
        masc, _ = segmentar(sam, proc, img, punto, [1])
        punto = punto[0]
        elegida = masc[int(np.argmax([acuerdo(x, m) for x in masc]))]
        fuera.append(not elegida[min(int(punto[1]), h - 1), min(int(punto[0]), w - 1)])

        variantes = {"sin limpiar": elegida}
        for modo, nombre in (("punto", "trozo del punto"), ("mayor", "trozo mayor"), ("grandes", "trozos >= 10 %")):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            variantes[nombre] = limpiar(elegida, punto, modo)
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1000)
        # limpiar las tres y elegir despues: el acuerdo con el mapa ya no lo ensucian los flecos
        limpias = [limpiar(x, punto, "punto") for x in masc]
        variantes["punto, limpiar y elegir"] = limpias[int(np.argmax([acuerdo(x, m) for x in limpias]))]
        variantes["techo sin limpiar"] = max(masc, key=lambda x: iou(caja_de(x), caja_de(real)))
        variantes["techo, trozo del punto"] = max(limpias, key=lambda x: iou(caja_de(x), caja_de(real)))

        for nombre, x in variantes.items():
            filas[nombre].append((iou(x, real), iou(caja_de(x), caja_de(real)), f["instancias"], f["area"]))
            if not nombre.startswith("techo"):
                n_tr[nombre].append(int((torch.unique(trozos(x)) > 0).sum()))

    n = len(filas["sin limpiar"])
    print(f"{a.sam}: mitad B, {n} imagenes, 1 punto, mascara elegida por acuerdo con el mapa")
    print(f"limpiar: {np.median(ms):.1f} ms de mediana (max {np.max(ms):.0f}); el punto cae fuera de la mascara en el {np.mean(fuera):.0%}\n")
    print(f"{'limpieza':26s} IoU masc.  IoU caja  cajas>=0,5  (1 inst. / varias)  (obj <3 % / >=3 %)  >5 trozos")
    for nombre, v in filas.items():
        v = np.array(v)
        b = v[:, 1] >= 0.5
        una, varias, peq = v[:, 2] == 1, v[:, 2] > 1, v[:, 3] < 0.03
        tr = f"{(np.array(n_tr[nombre]) > 5).mean():8.0%}" if nombre in n_tr else ""
        print(f"{nombre:26s} {v[:, 0].mean():8.3f}  {v[:, 1].mean():8.3f}  {b.mean():9.1%}"
              f"      ({b[una].mean():.0%} / {b[varias].mean():.0%})         ({b[peq].mean():.0%} / {b[~peq].mean():.0%})   {tr}")
    v0, v1 = np.array(filas["sin limpiar"]), np.array(filas["trozo del punto"])
    d = v1[:, 1] - v0[:, 1]
    print(f"\ntrozo del punto frente a sin limpiar, IoU de caja por imagen: mejora en {(d > 0.02).sum()}, "
          f"empeora en {(d < -0.02).sum()}, igual en {(np.abs(d) <= 0.02).sum()}")
    print(f"   cruzan el 0,5 hacia arriba {((v0[:, 1] < 0.5) & (v1[:, 1] >= 0.5)).sum()}, hacia abajo {((v0[:, 1] >= 0.5) & (v1[:, 1] < 0.5)).sum()}")


if __name__ == "__main__":
    main()
