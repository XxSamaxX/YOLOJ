#!/usr/bin/env python3
"""Una mascara por pico: cada maximo del mapa es un objeto aparte para SAM.

Un solo punto da una sola mascara, y con varias instancias el objetivo es la
union de todas. Aqui cada pico por encima de un umbral se pasa a SAM como objeto
propio (una sola llamada) y se unen las mascaras.

Cuantos picos y con que umbral se decide en la mitad A; la cifra es de la B.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from coco_atencion import COCO, mascara
from coco_mascara import mapa
from coco_sam import a_pixeles, acuerdo, caja_de, cargar_sam, iou

MAX_PICOS = (1, 2, 3, 5)
UMBRALES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SEPARACIONES = (2, 3)


def picos(m: np.ndarray, n: int, separacion: int) -> list[tuple[int, int, float]]:
    m = m.copy()
    salida = []
    for _ in range(n):
        y, x = np.unravel_index(m.argmax(), m.shape)
        if m[y, x] <= 0:
            break
        salida.append((int(y), int(x), float(m[y, x])))
        m[max(0, y - separacion):y + separacion + 1, max(0, x - separacion):x + separacion + 1] = -1
    return salida


@torch.inference_mode()
def mascaras_por_pico(sam, proc, img: Image.Image, puntos: list[list[float]]):
    """Una llamada, un objeto por punto. Devuelve mascaras [picos, 3, H, W] y puntuaciones [picos, 3]."""
    ent = proc(images=img, input_points=[[[p] for p in puntos]],
               input_labels=[[[1] for _ in puntos]], return_tensors="pt").to("cuda:0")
    ent["pixel_values"] = ent["pixel_values"].to(torch.bfloat16)
    sal = sam(**ent, multimask_output=True)
    masc = proc.post_process_masks(sal.pred_masks.float().cpu(), ent["original_sizes"])[0]
    return masc.numpy() > 0, sal.iou_scores.float().cpu().numpy()[0]


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
    for x in inst["annotations"]:
        anots[(x["image_id"], x["category_id"])].append(x)

    # resultados[mitad][(n, umbral, separacion, criterio)] = [(iou mascara, iou caja, instancias)]
    resultados: dict[str, dict] = {"A": defaultdict(list), "B": defaultdict(list)}
    for f in json.loads((datos / "indice.json").read_text()):
        z = {t: np.load(datos / f"{f['id']:012d}_{t}.npz") for t in ("pos", "gen")}
        m = mapa(z)
        img = Image.open(COCO / "val2017" / f["imagen"]).convert("RGB")
        w, h = img.size
        real = np.asarray(mascara(anots[(f["id"], ids[f["positiva"]])], w, h)) > 0
        if not real.any():
            continue
        for sep in SEPARACIONES:
            ps = picos(m, max(MAX_PICOS), sep)
            masc, punt = mascaras_por_pico(sam, proc, img, a_pixeles([(y, x) for y, x, _ in ps], m.shape, w, h))
            elegida = {
                "SAM": [masc[i, int(punt[i].argmax())] for i in range(len(ps))],
                "mapa": [masc[i, int(np.argmax([acuerdo(x, m) for x in masc[i]]))] for i in range(len(ps))],
            }
            for n, t, criterio in itertools.product(MAX_PICOS, UMBRALES, elegida):
                usar = [i for i, (_, _, v) in enumerate(ps[:n]) if i == 0 or v >= t]
                union = np.any([elegida[criterio][i] for i in usar], axis=0)
                resultados[f["mitad"]][(n, t, sep, criterio)].append(
                    (iou(union, real), iou(caja_de(union), caja_de(real)), f["instancias"]))

    def resumen(v) -> str:
        v = np.array(v)
        una, varias = v[v[:, 2] == 1], v[v[:, 2] > 1]
        return (f"IoU masc. {v[:, 0].mean():.3f}  cajas>=0,5 {(v[:, 1] >= 0.5).mean():5.1%}"
                f"  (1 instancia {(una[:, 1] >= 0.5).mean():.0%} / varias {(varias[:, 1] >= 0.5).mean():.0%})")

    base = (1, UMBRALES[0], SEPARACIONES[0], "mapa")
    mejor = max(resultados["A"], key=lambda k: np.mean([x[0] for x in resultados["A"][k]]))
    n, t, sep, criterio = mejor
    print(f"\n{a.sam}")
    print(f"elegido en A: hasta {n} picos, umbral {t} del maximo, separacion {sep} celdas, mascara segun {criterio}")
    print(f"   A, un pico      {resumen(resultados['A'][base])}")
    print(f"   A, elegido      {resumen(resultados['A'][mejor])}")
    print(f"   B, un pico      {resumen(resultados['B'][base])}")
    print(f"   B, elegido      {resumen(resultados['B'][mejor])}")
    print("\nen B, por numero maximo de picos (mismo umbral, separacion y criterio):")
    for k in MAX_PICOS:
        print(f"   hasta {k}: {resumen(resultados['B'][(k, t, sep, criterio)])}")


if __name__ == "__main__":
    main()
