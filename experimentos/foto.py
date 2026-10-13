#!/usr/bin/env python3
"""La cadena entera sobre una foto, con el resultado dibujado.

    foto.py foto.jpg                         las 80 clases de COCO
    foto.py foto.jpg -p person bottle        solo esas
    foto.py foto.jpg -p "Is anyone wearing glasses?"

Un prefijo de imagen y las preguntas en lote: el 4B decide que hay. Despues,
por cada "si":

    nombre de objeto      SAM 3 por texto -> todas sus instancias
    pregunta completa     mapa relativo de las cabezas de localizacion -> pico -> SAM -> mascara

Con --solo-punto todo va por el segundo camino, como antes.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from coco_atencion import COCO, pregunta
from coco_sam import UMBRAL, a_pixeles, acuerdo
from desgrane import GENERICO, cargar
from prefijo import Imagen

LOTE = 9


def cargar_sam(nombre: str):
    # SAM 3 trae dos modelos en el mismo repo; el que acepta puntos es el "tracker"
    if "sam3" in nombre:
        clase, clase_proc = transformers.Sam3TrackerModel, transformers.Sam3TrackerProcessor
    else:
        clase, clase_proc = transformers.Sam2Model, transformers.Sam2Processor
    return clase.from_pretrained(nombre, dtype=torch.bfloat16).to("cuda:0").eval(), clase_proc.from_pretrained(nombre)


def mapa_relativo(aten: np.ndarray, gen: np.ndarray) -> np.ndarray:
    """aten, gen: [cabezas, gh, gw] -> mapa en [0, 1]."""
    rel = aten / (gen + 1e-4)
    m = (rel / (rel.sum((1, 2), keepdims=True) + 1e-9)).sum(0)
    p = np.pad(m, 1, mode="edge")
    m = sum(p[i:i + m.shape[0], j:j + m.shape[1]] for i in range(3) for j in range(3)) / 9
    return m / m.max()


@torch.inference_mode()
def mascara_sam(sam, proc, img: Image.Image, m: np.ndarray):
    y, x = np.unravel_index(m.argmax(), m.shape)
    punto = a_pixeles([(y, x)], m.shape, *img.size)
    ent = proc(images=img, input_points=[[punto]], input_labels=[[[1]]], return_tensors="pt").to("cuda:0")
    ent["pixel_values"] = ent["pixel_values"].to(torch.bfloat16)
    sal = sam(**ent, multimask_output=True)
    masc = proc.post_process_masks(sal.pred_masks.float().cpu(), ent["original_sizes"])[0][0].numpy() > 0
    return masc[int(np.argmax([acuerdo(x, m) for x in masc]))], punto[0]


class Sam3Texto:
    """SAM 3 por concepto: la imagen se codifica una vez y cada nombre es un paso corto."""

    def __init__(self):
        self.sam = transformers.Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to("cuda:0").eval()
        self.proc = transformers.Sam3Processor.from_pretrained("facebook/sam3")

    @torch.inference_mode()
    def imagen(self, img: Image.Image) -> None:
        ent = self.proc(images=img, return_tensors="pt").to("cuda:0")
        self.vision = self.sam.get_vision_features(pixel_values=ent["pixel_values"].to(torch.bfloat16))
        self.medida = [img.size[::-1]]

    @torch.inference_mode()
    def buscar(self, nombre: str, corte: float) -> list[tuple[np.ndarray, float]]:
        ent = self.proc(text=nombre, return_tensors="pt").to("cuda:0")
        sal = self.sam(vision_embeds=self.vision, input_ids=ent["input_ids"], attention_mask=ent["attention_mask"])
        for k, v in sal.items():
            if torch.is_tensor(v) and v.is_floating_point():
                sal[k] = v.float()
        r = self.proc.post_process_instance_segmentation(sal, threshold=corte, mask_threshold=0.5,
                                                         target_sizes=self.medida)[0]
        return [(m.cpu().numpy() > 0, float(s)) for m, s in zip(r["masks"], r["scores"]) if m.any()]


def centro(masc: np.ndarray) -> list[float]:
    """Un punto dentro de la mascara donde colgar la etiqueta: el pixel mas cercano a su centro de masas."""
    ys, xs = np.nonzero(masc)
    k = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)
    return [float(xs[k]), float(ys[k])]


def dibujar(img: Image.Image, hallazgos: list[dict]) -> Image.Image:
    lienzo = img.convert("RGBA")
    try:
        fuente = ImageFont.load_default(size=max(14, img.width // 55))
    except TypeError:
        fuente = ImageFont.load_default()
    for i, h in enumerate(hallazgos):
        color = tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i * 0.618 % 1, 0.85, 1.0))
        m = Image.fromarray(h["mascara"]).convert("L")
        capa = Image.new("RGBA", img.size, color + (0,))
        capa.putalpha(m.point(lambda v: 95 if v else 0))
        lienzo = Image.alpha_composite(lienzo, capa)
        borde = m.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
        linea = Image.new("RGBA", img.size, color + (0,))
        linea.putalpha(borde.point(lambda v: 255 if v else 0))
        lienzo = Image.alpha_composite(lienzo, linea)
        h["color"] = color
    d = ImageDraw.Draw(lienzo)
    for h in hallazgos:
        x, y = h["punto"]
        r = max(4, img.width // 200)
        d.ellipse([x - r, y - r, x + r, y + r], fill=h["color"] + (255,), outline=(255, 255, 255, 255), width=2)
        d.text((x + r + 4, y), h.get("rotulo") or f"{h['nombre']} {h['gap']:+.0f}", font=fuente, anchor="lm",
               fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))
    return lienzo.convert("RGB")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("imagen")
    p.add_argument("-p", "--pregunta", nargs="+", help="nombres de objeto o preguntas completas; sin esto, COCO")
    p.add_argument("--sam", default="facebook/sam2.1-hiera-small", help="el SAM que recibe puntos")
    p.add_argument("--solo-punto", action="store_true", help="no usar SAM 3 por texto")
    p.add_argument("--corte-sam3", type=float, default=0.5, help="puntuacion minima de una instancia de SAM 3")
    p.add_argument("--lado", type=int, default=960, help="lado largo al que se reduce la foto para el 4B")
    p.add_argument("--corte", type=float, default=0.0, help="gap minimo para dar un objeto por presente")
    p.add_argument("--lote", type=int, help="preguntas por lote; por defecto 9, o 4 con SAM 3 cargado (no cabe mas)")
    p.add_argument("--salida")
    a = p.parse_args()

    if a.pregunta:
        nombres = a.pregunta
    else:
        nombres = [c["name"] for c in json.loads((COCO / "instances_val2017.json").read_text())["categories"]]
    textos = [n if n.rstrip().endswith("?") else pregunta(n) for n in nombres]

    original = Image.open(a.imagen).convert("RGB")
    f = min(1.0, a.lado / max(original.size))
    img = original.resize((round(original.width * f), round(original.height * f)), Image.LANCZOS)

    modelo, proc = cargar(atenciones=True)
    sam, proc_sam = cargar_sam(a.sam)
    sam3 = None if a.solo_punto else Sam3Texto()
    print(f"VRAM con los modelos cargados: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    lote = a.lote or (4 if sam3 else LOTE)

    def ahora() -> float:
        torch.cuda.synchronize()
        return time.perf_counter()

    for vuelta in range(2):                    # la primera calienta la grafica; se cuenta la segunda
        t0 = ahora()
        pre = Imagen(modelo, proc, img)
        t1 = ahora()
        gaps, atens = [], []
        todos = textos + [GENERICO]
        for i in range(0, len(todos), lote):
            g, at = pre.preguntar(todos[i:i + lote])
            gaps.extend(g)
            atens.extend(at)
        t2 = ahora()
        hallazgos, t_vision, codificada = [], 0.0, False
        for nombre, gap, at in zip(nombres, gaps, atens):
            if gap <= a.corte:
                continue
            instancias = []
            if sam3 and not nombre.rstrip().endswith("?"):
                if not codificada:
                    ta = ahora()
                    sam3.imagen(original)
                    t_vision, codificada = ahora() - ta, True
                instancias = sam3.buscar(nombre, a.corte_sam3)
            for k, (masc, punt) in enumerate(instancias):
                rotulo = f"{nombre} {gap:+.0f}" if len(instancias) == 1 else f"{nombre} {k + 1}"
                hallazgos.append({"nombre": nombre, "gap": float(gap), "mascara": masc, "punto": centro(masc),
                                  "via": f"SAM 3 texto {punt:.2f}", "rotulo": rotulo})
            if not instancias:            # pregunta libre, o un nombre que SAM 3 no encuentra
                m = mapa_relativo(at, atens[-1])
                masc, punto = mascara_sam(sam, proc_sam, original, m)
                hallazgos.append({"nombre": nombre, "gap": float(gap), "mascara": masc, "punto": punto,
                                  "via": "atencion + punto"})
        t3 = ahora()

    print(f"\nfoto {original.size[0]}x{original.size[1]} -> {img.size[0]}x{img.size[1]} para el 4B, "
          f"rejilla {pre.rejilla[0]}x{pre.rejilla[1]} ({len(pre.pos_img)} tokens de imagen)")
    print(f"   prefijo de la imagen            {(t1 - t0) * 1000:6.0f} ms")
    print(f"   {len(todos):2d} preguntas en lotes de {lote}      {(t2 - t1) * 1000:6.0f} ms")
    print(f"   SAM, {len(hallazgos)} mascaras               {(t3 - t2) * 1000:6.0f} ms"
          + (f"   (de ellos {t_vision * 1000:.0f} ms codificar la imagen en SAM 3)" if t_vision else ""))
    print(f"   total                           {(t3 - t0) * 1000:6.0f} ms   (pico de VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)")
    print("\npresentes (gap > corte):")
    for h in sorted(hallazgos, key=lambda h: -h["gap"]):
        print(f"   {h['nombre']:22s} {h['gap']:+6.1f}   mascara: {h['mascara'].mean():5.1%} de la imagen   {h['via']}")
    dudosos = sorted(((g, n) for n, g in zip(nombres, gaps) if -6 < g <= a.corte), reverse=True)[:8]
    if dudosos:
        print("ausentes por poco: " + ", ".join(f"{n} {g:+.1f}" for g, n in dudosos))

    salida = Path(a.salida or Path(a.imagen).with_name(Path(a.imagen).stem + "__yoloj.jpg"))
    dibujar(original, hallazgos).save(salida, quality=90)
    print(salida)


if __name__ == "__main__":
    main()
