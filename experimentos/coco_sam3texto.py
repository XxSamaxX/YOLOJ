#!/usr/bin/env python3
"""SAM 3 pidiendole el objeto por su nombre, frente a la cadena de atencion + punto.

SAM 3 acepta un concepto en texto y devuelve todas sus instancias, cada una con
su mascara, su caja y su puntuacion. Aqui se mide en la mitad B, con las mismas
imagenes y la misma referencia que coco_sam.py:

    union        todas las mascaras juntas contra la union de las instancias reales
    instancias   cuantas instancias reales quedan cubiertas por una caja propia (IoU >= 0,5)
    presencia    separa la categoria que esta de una que no esta?

La cadena de atencion + punto se mide al lado con la misma vara.
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
import transformers
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from coco_atencion import COCO, mascara
from coco_mascara import mapa
from coco_sam import a_pixeles, acuerdo, caja_de, cargar_sam, iou, picos, segmentar

CORTES = (0.3, 0.5, 0.7)


@torch.inference_mode()
def por_texto(sam, proc, img: Image.Image, texto: str, minimo: float = 0.05):
    """Devuelve (mascaras [N, H, W] bool, cajas [N, 4] xyxy, puntuaciones [N]) de todas las instancias."""
    ent = proc(images=img, text=texto, return_tensors="pt").to("cuda:0")
    ent["pixel_values"] = ent["pixel_values"].to(sam.dtype)
    sal = sam(**ent)
    for k, v in sal.items():
        if torch.is_tensor(v) and v.is_floating_point():
            sal[k] = v.float()
    r = proc.post_process_instance_segmentation(sal, threshold=minimo, mask_threshold=0.5,
                                                target_sizes=[img.size[::-1]])[0]
    return r["masks"].cpu().numpy() > 0, r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy()


def iou_cajas(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a [N, 4], b [M, 4] en xyxy -> [N, M]."""
    x0, y0 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x1, y1 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda c: (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def casar(pred: np.ndarray, real: np.ndarray) -> int:
    """Cuantas cajas reales quedan emparejadas, una a una, con una predicha a IoU >= 0,5."""
    if not len(pred) or not len(real):
        return 0
    m = iou_cajas(pred, real)
    n = 0
    while m.size and m.max() >= 0.5:
        i, j = np.unravel_index(m.argmax(), m.shape)
        m[i, :], m[:, j] = -1, -1
        n += 1
    return n


def caja_xyxy(masc: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(masc)
    return np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], float) if len(ys) else np.zeros((0, 4))


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--datos", default="out/coco")
    p.add_argument("--base", default="facebook/sam2.1-hiera-large", help="SAM de la cadena de atencion + punto")
    a = p.parse_args()
    datos = Path(a.datos)
    sam3 = transformers.Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to("cuda:0").eval()
    proc3 = transformers.Sam3Processor.from_pretrained("facebook/sam3")
    base, proc_base = cargar_sam(a.base)

    inst = json.loads((COCO / "instances_val2017.json").read_text())
    ids = {c["name"]: c["id"] for c in inst["categories"]}
    anots = defaultdict(list)
    for an in inst["annotations"]:
        anots[(an["image_id"], an["category_id"])].append(an)

    filas = defaultdict(list)          # (IoU masc., IoU caja, instancias, area)
    cuenta = defaultdict(lambda: np.zeros(3))   # reales casadas, reales, predichas
    por_n = defaultdict(lambda: np.zeros(2))
    presencia = {"pos": [], "neg": []}
    falsas = defaultdict(int)
    ms = []
    for f in json.loads((datos / "indice.json").read_text()):
        if f["mitad"] != "B":
            continue
        img = Image.open(COCO / "val2017" / f["imagen"]).convert("RGB")
        w, h = img.size
        suyas = anots[(f["id"], ids[f["positiva"]])]
        real = np.asarray(mascara(suyas, w, h)) > 0
        if not real.any():
            continue
        cajas_reales = np.array([[x, y, x + bw, y + bh] for x, y, bw, bh in
                                 (s["bbox"] for s in suyas if not s.get("iscrowd"))], float).reshape(-1, 4)
        varias = "varias" if f["instancias"] > 1 else "una"

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        masc, cajas, punt = por_texto(sam3, proc3, img, f["positiva"])
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
        presencia["pos"].append(punt.max() if len(punt) else 0.0)
        for c in CORTES:
            q = punt >= c
            union = masc[q].any(0) if q.any() else np.zeros_like(real)
            nombre = f"SAM 3 por texto, >= {c}"
            filas[nombre].append((iou(union, real), iou(caja_de(union), caja_de(real)), f["instancias"], f["area"]))
            v = np.array([casar(cajas[q], cajas_reales), len(cajas_reales), q.sum()])
            cuenta[nombre] += v
            por_n[(nombre, varias)] += v[:2]

        _, _, punt_neg = por_texto(sam3, proc3, img, f["negativa"])
        presencia["neg"].append(punt_neg.max() if len(punt_neg) else 0.0)
        for c in CORTES:
            falsas[c] += int((punt_neg >= c).any())

        # la cadena de siempre: pico del mapa -> punto -> la mascara que mas coincide con el mapa
        m = mapa({t: np.load(datos / f"{f['id']:012d}_{t}.npz") for t in ("pos", "gen")})
        tres, _ = segmentar(base, proc_base, img, a_pixeles(picos(m, 1), m.shape, w, h), [1])
        una = tres[int(np.argmax([acuerdo(x, m) for x in tres]))]
        nombre = "atencion + punto"
        filas[nombre].append((iou(una, real), iou(caja_de(una), caja_de(real)), f["instancias"], f["area"]))
        v = np.array([casar(caja_xyxy(una), cajas_reales), len(cajas_reales), 1])
        cuenta[nombre] += v
        por_n[(nombre, varias)] += v[:2]

    n = len(filas["atencion + punto"])
    print(f"mitad B, {n} imagenes. SAM 3 por texto: {np.median(ms):.0f} ms por imagen y concepto "
          f"(pico de VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB con {a.base.split('/')[-1]} al lado)\n")
    print("UNION de las mascaras contra la union real")
    print(f"{'metodo':26s} IoU masc.  IoU caja  cajas>=0,5  (1 inst. / varias)  (obj <3 % / >=3 %)")
    for nombre in ["atencion + punto"] + [f"SAM 3 por texto, >= {c}" for c in CORTES]:
        v = np.array(filas[nombre])
        b = v[:, 1] >= 0.5
        una, varias, peq = v[:, 2] == 1, v[:, 2] > 1, v[:, 3] < 0.03
        print(f"{nombre:26s} {v[:, 0].mean():8.3f}  {v[:, 1].mean():8.3f}  {b.mean():9.1%}"
              f"      ({b[una].mean():.0%} / {b[varias].mean():.0%})         ({b[peq].mean():.0%} / {b[~peq].mean():.0%})")
    print("\nINSTANCIAS: cada caja real con una caja predicha propia a IoU >= 0,5")
    print(f"{'metodo':26s} encontradas          (1 inst. / varias)   cajas dadas  de ellas buenas")
    for nombre, (ok, reales, dadas) in cuenta.items():
        u, v = por_n[(nombre, "una")], por_n[(nombre, "varias")]
        print(f"{nombre:26s} {int(ok):4d} de {int(reales):4d} = {ok / reales:5.1%}   ({u[0] / u[1]:.0%} / {v[0] / v[1]:.0%})"
              f"        {int(dadas):6d}       {ok / max(dadas, 1):6.1%}")
    print(f"\nPRESENCIA con la puntuacion maxima de SAM 3 (categoria presente contra una ausente al azar): "
          f"AUC {auc(presencia['pos'], presencia['neg']):.3f}")
    for c in CORTES:
        print(f"   corte {c}: no encuentra la presente en {np.mean(np.array(presencia['pos']) < c):5.1%}, "
              f"inventa la ausente en {falsas[c] / n:5.1%}")


if __name__ == "__main__":
    main()
