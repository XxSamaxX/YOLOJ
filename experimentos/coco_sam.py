#!/usr/bin/env python3
"""De la atencion a una mascara: los picos del mapa como indicaciones para SAM 2.1.

Calidad: usa los mapas que guardo coco_atencion.py (mitad B, cabezas elegidas en
A), asi que solo carga SAM. Compara varias formas de indicarle el objeto y mide,
al pixel, contra la mascara real de COCO.

Con --tiempos carga ademas Qwen3-VL-4B al lado y cronometra la cadena entera:
cabe o no cabe en la grafica, y cuanto cuesta cada etapa.
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
from coco_atencion import COCO, mascara, pregunta
from coco_mascara import CABEZAS, mapa

UMBRAL = 0.5          # del maximo del mapa; elegido en la mitad A por coco_mascara.py


def cargar_sam(nombre: str, dtype=torch.bfloat16):
    # SAM 3 trae dos modelos en el mismo repo; el que acepta puntos es el "tracker"
    if "sam3" in nombre:
        clase, clase_proc = transformers.Sam3TrackerModel, transformers.Sam3TrackerProcessor
    else:
        clase, clase_proc = transformers.Sam2Model, transformers.Sam2Processor
    return clase.from_pretrained(nombre, dtype=dtype).to("cuda:0").eval(), clase_proc.from_pretrained(nombre)


def picos(m: np.ndarray, n: int, separacion: int = 3) -> list[tuple[int, int]]:
    """Los n maximos del mapa, a mas de `separacion` celdas unos de otros y por encima del umbral."""
    m = m.copy()
    salida = []
    for _ in range(n):
        y, x = np.unravel_index(m.argmax(), m.shape)
        if m[y, x] < UMBRAL and salida:
            break
        salida.append((int(y), int(x)))
        m[max(0, y - separacion):y + separacion + 1, max(0, x - separacion):x + separacion + 1] = -1
    return salida


def a_pixeles(celdas, forma, ancho, alto) -> list[list[float]]:
    gh, gw = forma
    return [[(x + 0.5) / gw * ancho, (y + 0.5) / gh * alto] for y, x in celdas]


def caja_del_mapa(m: np.ndarray, ancho: int, alto: int) -> list[float]:
    ys, xs = np.nonzero(m >= UMBRAL)
    gh, gw = m.shape
    return [xs.min() / gw * ancho, ys.min() / gh * alto, (xs.max() + 1) / gw * ancho, (ys.max() + 1) / gh * alto]


@torch.inference_mode()
def segmentar(sam, proc, img: Image.Image, puntos=None, etiquetas=None, caja=None):
    """Devuelve (mascaras [3, H, W] bool, puntuaciones [3]) de un objeto."""
    extra = {}
    if puntos is not None:
        extra["input_points"] = [[puntos]]
        extra["input_labels"] = [[etiquetas]]
    if caja is not None:
        extra["input_boxes"] = [[caja]]
    ent = proc(images=img, return_tensors="pt", **extra).to("cuda:0")
    ent["pixel_values"] = ent["pixel_values"].to(sam.dtype)
    sal = sam(**ent, multimask_output=True)
    masc = proc.post_process_masks(sal.pred_masks.float().cpu(), ent["original_sizes"])[0][0]
    return masc.numpy() > 0, sal.iou_scores.float().cpu().numpy()[0, 0]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def caja_de(m: np.ndarray) -> np.ndarray:
    c = np.zeros_like(m)
    ys, xs = np.nonzero(m)
    if len(ys):
        c[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return c


def n_trozos(masc: np.ndarray, lado: int = 256) -> int:
    """Componentes conexas de una mascara, a baja resolucion.

    Cada pixel empieza con su numero y en cada paso toma el mayor de sus vecinos
    dentro de la mascara; al estabilizarse queda un numero por trozo.
    """
    t = torch.from_numpy(masc).to("cuda:0")[None, None].float()
    f = lado / max(masc.shape)
    t = torch.nn.functional.interpolate(t, size=(round(masc.shape[0] * f), round(masc.shape[1] * f)), mode="nearest")
    ids = (torch.arange(t.numel(), device=t.device).view_as(t) + 1).float() * t
    while True:
        nuevo = torch.nn.functional.max_pool2d(ids, 3, 1, 1) * t
        if torch.equal(nuevo, ids):
            break
        ids = nuevo
    return int((torch.unique(ids) > 0).sum())


def acuerdo(masc: np.ndarray, m: np.ndarray) -> float:
    """Cuanto coincide una mascara con el mapa de atencion umbralizado, en la rejilla."""
    gh, gw = m.shape
    red = np.asarray(Image.fromarray(masc).resize((gw, gh), Image.BOX).convert("L")) > 63
    return iou(red, m >= UMBRAL)


def calidad(sam, proc, datos: Path) -> None:
    inst = json.loads((COCO / "instances_val2017.json").read_text())
    ids = {c["name"]: c["id"] for c in inst["categories"]}
    anots = defaultdict(list)
    for a in inst["annotations"]:
        anots[(a["image_id"], a["category_id"])].append(a)

    filas = defaultdict(list)
    tiempos, trozos = [], []
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
        p1 = a_pixeles(picos(m, 1), m.shape, w, h)
        p3 = a_pixeles(picos(m, 3), m.shape, w, h)
        caja = caja_del_mapa(m, w, h)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        formas = {
            "1 punto": segmentar(sam, proc, img, p1, [1]),
        }
        torch.cuda.synchronize()
        tiempos.append((time.perf_counter() - t0) * 1000)
        formas["hasta 3 puntos"] = segmentar(sam, proc, img, p3, [1] * len(p3))
        formas["caja del mapa"] = segmentar(sam, proc, img, caja=caja)
        formas["caja + 1 punto"] = segmentar(sam, proc, img, p1, [1], caja=caja)

        # en cuantos trozos sueltos sale la mascara: un borde deshilachado deja muchos
        elegida = formas["1 punto"][0][int(np.argmax([acuerdo(x, m) for x in formas["1 punto"][0]]))]
        trozos.append(n_trozos(elegida))
        for nombre, (masc, punt) in formas.items():
            for criterio, k in (("la que SAM puntua mas", int(punt.argmax())),
                                ("la que mas coincide con el mapa", int(np.argmax([acuerdo(x, m) for x in masc])))):
                filas[(nombre, criterio)].append(
                    (iou(masc[k], real), iou(caja_de(masc[k]), caja_de(real)), f["instancias"], f["area"]))
            filas[(nombre, "techo: la mejor de las 3")].append(
                (max(iou(x, real) for x in masc), max(iou(caja_de(x), caja_de(real)) for x in masc),
                 f["instancias"], f["area"]))

    n = len(next(iter(filas.values())))
    print(f"mitad B, {n} imagenes, al pixel. SAM: {np.median(tiempos):.0f} ms por imagen (codificar + 1 indicacion)")
    print(f"trozos sueltos por mascara (1 punto, elegida por el mapa): mediana {np.median(trozos):.0f}, media {np.mean(trozos):.1f}, "
          f"con mas de 5 trozos: {(np.array(trozos) > 5).mean():.0%}\n")
    print(f"{'indicacion':16s} {'mascara elegida':32s} IoU masc.  IoU caja  cajas>=0,5  (1 instancia / varias)")
    for (nombre, criterio), v in filas.items():
        v = np.array(v)
        una, varias = v[v[:, 2] == 1], v[v[:, 2] > 1]
        print(f"{nombre:16s} {criterio:32s} {v[:, 0].mean():8.3f}  {v[:, 1].mean():8.3f}  {(v[:, 1] >= 0.5).mean():9.1%}"
              f"   ({(una[:, 1] >= 0.5).mean():.0%} / {(varias[:, 1] >= 0.5).mean():.0%})")
    v = np.array(filas[("caja + 1 punto", "la que SAM puntua mas")])
    for nombre, filtro in (("objeto < 3 % de la imagen", v[:, 3] < 0.03), ("objeto >= 3 %", v[:, 3] >= 0.03)):
        s = v[filtro]
        print(f"   caja + 1 punto, {nombre:26s} n={len(s):3d}  IoU masc. {s[:, 0].mean():.3f}  cajas>=0,5 {(s[:, 1] >= 0.5).mean():.0%}")


@torch.inference_mode()
def cadena(sam, proc_sam, n: int = 20) -> None:
    """La cadena entera con los dos modelos en la grafica: memoria y ms por etapa."""
    from desgrane import GENERICO, cargar, entradas

    vlm, proc = cargar(atenciones=True)
    indice = json.loads((Path("out/coco") / "indice.json").read_text())[:n + 2]
    etapas = defaultdict(list)
    capas = sorted({c for c, _ in CABEZAS})

    def atencion(img, texto):
        ent = entradas(proc, img, texto).to(vlm.device)
        sal = vlm(**ent, output_attentions=True, use_cache=False, return_dict=True)
        pos = (ent["input_ids"][0] == vlm.config.image_token_id).nonzero().squeeze(1)
        _, h, w = ent["image_grid_thw"][0].tolist()
        a = {c: sal.attentions[c][0, :, -1, pos].float().reshape(-1, h // 2, w // 2) for c in capas}
        return a, sal.logits[0, -1]

    torch.cuda.reset_peak_memory_stats()
    for f in indice:
        img = Image.open(COCO / "val2017" / f["imagen"]).convert("RGB")
        w, h = img.size
        marcas = [time.perf_counter()]

        def marca():
            torch.cuda.synchronize()
            marcas.append(time.perf_counter())

        a_pos, _ = atencion(img, pregunta(f["positiva"])); marca()
        a_gen, _ = atencion(img, GENERICO); marca()
        rel = [a_pos[c][hh] / (a_gen[c][hh] + 1e-4) for c, hh in CABEZAS]
        m = sum(r / (r.sum() + 1e-9) for r in rel).cpu().numpy()
        m = m / m.max()
        p1 = a_pixeles(picos(m, 1), m.shape, w, h)
        segmentar(sam, proc_sam, img, p1, [1], caja=caja_del_mapa(m, w, h)); marca()
        for nombre, dt in zip(("pregunta (4B)", "generica (4B)", "SAM"), np.diff(marcas) * 1000):
            etapas[nombre].append(dt)
    print(f"\nlos dos modelos a la vez: pico de {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB de VRAM")
    total = 0.0
    for nombre, v in etapas.items():
        med = float(np.median(v[2:]))
        total += med
        print(f"   {nombre:16s} {med:6.0f} ms")
    print(f"   {'total':16s} {total:6.0f} ms por imagen y predicado  ({1000 / total:.1f} por segundo)")
    print(f"   cada predicado mas sobre la misma imagen: {np.median(etapas['pregunta (4B)'][2:]) + np.median(etapas['SAM'][2:]):.0f} ms")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sam", default="facebook/sam2.1-hiera-small")
    p.add_argument("--datos", default="out/coco")
    p.add_argument("--tiempos", action="store_true")
    p.add_argument("--fp32", action="store_true", help="SAM en 32 bits en vez de bf16")
    a = p.parse_args()
    sam, proc = cargar_sam(a.sam, torch.float32 if a.fp32 else torch.bfloat16)
    print("32 bits" if a.fp32 else "bf16")
    print(f"{a.sam}: {sum(x.numel() for x in sam.parameters()) / 1e6:.0f} M de parametros, "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    if a.tiempos:
        cadena(sam, proc)
    else:
        calidad(sam, proc, Path(a.datos))


if __name__ == "__main__":
    main()
