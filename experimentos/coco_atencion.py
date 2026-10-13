#!/usr/bin/env python3
"""Atencion de Qwen3-VL-4B sobre COCO val2017, con la mascara real de cada objeto.

Por cada imagen muestreada, tres forwards:
  positivo   "Is there a <categoria presente>?"   con su mascara en la rejilla
  negativo   "Is there a <categoria ausente>?"
  generico   un predicado sin contenido, para localizar sumideros

Guarda un .npz por forward. El analisis (coco_cabezas.py) ya no necesita la GPU.

Las imagenes se reparten en dos mitades, A y B: las cabezas se eligen en A y se
miden en B, para no repetir la circularidad del primer desgrane.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from desgrane import GENERICO, cargar, desgranar

COCO = Path(__file__).resolve().parent.parent / "data" / "coco"
AREA_MIN, AREA_MAX = 0.005, 0.30      # fraccion de imagen que ocupa la categoria


def mascara(anotaciones: list[dict], ancho: int, alto: int) -> Image.Image:
    m = Image.new("L", (ancho, alto), 0)
    d = ImageDraw.Draw(m)
    for a in anotaciones:
        for poligono in a["segmentation"]:
            if len(poligono) >= 6:
                d.polygon(poligono, fill=255)
    return m


def muestrear(n: int, semilla: int) -> list[dict]:
    inst = json.loads((COCO / "instances_val2017.json").read_text())
    nombres = {c["id"]: c["name"] for c in inst["categories"]}
    imagenes = {i["id"]: i for i in inst["images"]}
    por_imagen: dict[int, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for a in inst["annotations"]:
        por_imagen[a["image_id"]][a["category_id"]].append(a)

    rng = random.Random(semilla)
    ids = sorted(por_imagen)
    rng.shuffle(ids)
    casos = []
    for iid in ids:
        im = imagenes[iid]
        candidatas = []
        for cid, anots in por_imagen[iid].items():
            if any(a["iscrowd"] for a in anots):        # las aglomeraciones van en RLE y sin instancias
                continue
            frac = np.asarray(mascara(anots, im["width"], im["height"])).mean() / 255
            if AREA_MIN <= frac <= AREA_MAX:
                candidatas.append((cid, frac))
        if not candidatas:
            continue
        cid, frac = rng.choice(sorted(candidatas))
        ausente = rng.choice(sorted(set(nombres) - set(por_imagen[iid])))
        casos.append({
            "imagen": im["file_name"], "id": iid,
            "mitad": "A" if len(casos) < n // 2 else "B",
            "positiva": nombres[cid], "area": frac, "instancias": len(por_imagen[iid][cid]),
            "negativa": nombres[ausente],
            "_anots": por_imagen[iid][cid], "_wh": (im["width"], im["height"]),
        })
        if len(casos) == n:
            break
    return casos


def pregunta(categoria: str) -> str:
    return f"Is there {'an' if categoria[0] in 'aeiou' else 'a'} {categoria} in the image?"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--n", type=int, default=400, help="imagenes (mitad A, mitad B)")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--salida", default="out/coco")
    a = p.parse_args()
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)

    casos = muestrear(a.n, a.semilla)
    modelo, proc = cargar(atenciones=True)
    indice = []
    for k, c in enumerate(casos):
        img = Image.open(COCO / "val2017" / c["imagen"]).convert("RGB")
        ficha = {x: v for x, v in c.items() if not x.startswith("_")}
        for tipo, texto in (("pos", pregunta(c["positiva"])), ("neg", pregunta(c["negativa"])),
                            ("gen", GENERICO)):
            r = desgranar(modelo, proc, img, texto, None)
            extra = {}
            if tipo == "pos":
                gh, gw = r["rejilla"]
                # fraccion de cada celda de la rejilla cubierta por el objeto
                m = mascara(c["_anots"], *c["_wh"]).resize((gw, gh), Image.BOX)
                extra["mascara"] = np.asarray(m, np.float32) / 255
            np.savez_compressed(
                salida / f"{c['id']:012d}_{tipo}.npz",
                aten=r["aten"].astype(np.float16), gap=r["gap"],
                gap_capa=r["gap_capa"].astype(np.float32), **extra)
            ficha[f"gap_{tipo}"] = r["gap"]
        indice.append(ficha)
        if k % 25 == 0:
            print(f"{k:4d}/{len(casos)}  {c['imagen']}  +{c['positiva']} {ficha['gap_pos']:+.1f}"
                  f"  -{c['negativa']} {ficha['gap_neg']:+.1f}", flush=True)
    (salida / "indice.json").write_text(json.dumps(indice, indent=1))
    print(f"{len(indice)} imagenes en {salida}")


if __name__ == "__main__":
    main()
