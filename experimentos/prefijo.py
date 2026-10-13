#!/usr/bin/env python3
"""La imagen se procesa una vez; las preguntas van en lote detras.

Todas las preguntas sobre una imagen comparten el principio del prompt: sistema
+ imagen. Ese prefijo (~340 de los ~385 tokens) se calcula una sola vez y se
guarda su cache de claves y valores. Cada pregunta es entonces un sufijo de ~45
tokens, y N preguntas son un lote de N sufijos sobre la misma cache.

Comprueba que da lo mismo que el forward normal (gap y mapas) y cronometra.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from coco_atencion import COCO, pregunta
from coco_mascara import CABEZAS
from desgrane import GENERICO, SISTEMA, cargar, entradas

COLA = "<|im_end|>\n<|im_start|>assistant\n"
CAPAS = sorted({c for c, _ in CABEZAS})


def _json(texto: str) -> str:
    return json.dumps({"criterion": texto, "options": [{"letter": "A", "description": "yes"},
                                                       {"letter": "B", "description": "no"}]})


class Imagen:
    """El prefijo de una imagen ya pasado por el modelo, listo para recibir preguntas."""

    @torch.inference_mode()
    def __init__(self, modelo, proc, img: Image.Image):
        self.modelo, self.tok = modelo, proc.tokenizer
        ent = entradas(proc, img, GENERICO).to(modelo.device)
        ids = ent["input_ids"][0]
        # el prefijo acaba en <|vision_end|>: un token especial, asi que el corte es limpio
        self.n = int((ids == modelo.config.vision_end_token_id).nonzero()[0, 0]) + 1
        sufijo = self.tok.encode(_json(GENERICO) + COLA, add_special_tokens=False)
        assert ids[self.n:].tolist() == sufijo, "el prompt no se parte donde se esperaba"
        self.pos_img = (ids[:self.n] == modelo.config.image_token_id).nonzero().squeeze(1)
        _, h, w = ent["image_grid_thw"][0].tolist()
        self.rejilla = (h // 2, w // 2)
        sal = modelo(input_ids=ent["input_ids"][:, :self.n], attention_mask=ent["attention_mask"][:, :self.n],
                     mm_token_type_ids=ent["mm_token_type_ids"][:, :self.n],
                     pixel_values=ent["pixel_values"], image_grid_thw=ent["image_grid_thw"],
                     use_cache=True, return_dict=True)
        self.cache = sal.past_key_values
        # desfase entre posicion en la secuencia y posicion rotatoria: la imagen ocupa menos de lo que mide
        self.delta = int(modelo.model.rope_deltas[0])
        self.a, self.b = (self.tok.encode(x, add_special_tokens=False)[0] for x in "AB")

    @torch.inference_mode()
    def preguntar(self, textos: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (gaps [N], atencion [N, cabezas elegidas, gh, gw]) de un lote de preguntas."""
        sufijos = [self.tok.encode(_json(t) + COLA, add_special_tokens=False) for t in textos]
        largo = max(map(len, sufijos))
        n_lote = len(sufijos)
        # relleno a la derecha: como la atencion es causal, nadie llega a mirarlo
        ids = torch.full((n_lote, largo), self.tok.pad_token_id, device=self.modelo.device)
        for i, s in enumerate(sufijos):
            ids[i, :len(s)] = torch.tensor(s)
        ultimo = torch.tensor([len(s) - 1 for s in sufijos], device=ids.device)
        pos = (self.n + self.delta + torch.arange(largo, device=ids.device)).view(1, 1, -1).expand(3, n_lote, -1)
        cache = copy.deepcopy(self.cache)
        cache.batch_repeat_interleave(n_lote)
        sal = self.modelo(input_ids=ids, position_ids=pos, past_key_values=cache,
                          attention_mask=torch.ones(n_lote, self.n + largo, dtype=torch.long, device=ids.device),
                          output_attentions=True, use_cache=True, return_dict=True)
        fila = torch.arange(n_lote, device=ids.device)
        logits = sal.logits[fila, ultimo].float()
        gaps = (logits[:, self.a] - logits[:, self.b]).cpu().numpy()
        aten = torch.stack([sal.attentions[c][fila, h, ultimo][:, self.pos_img] for c, h in CABEZAS], 1)
        return gaps, aten.float().reshape(n_lote, len(CABEZAS), *self.rejilla).cpu().numpy()


@torch.inference_mode()
def normal(modelo, proc, img, texto):
    """El forward entero de siempre, para comparar."""
    ent = entradas(proc, img, texto).to(modelo.device)
    sal = modelo(**ent, output_attentions=True, use_cache=False, return_dict=True)
    pos = (ent["input_ids"][0] == modelo.config.image_token_id).nonzero().squeeze(1)
    _, h, w = ent["image_grid_thw"][0].tolist()
    a, b = (proc.tokenizer.encode(x, add_special_tokens=False)[0] for x in "AB")
    aten = torch.stack([sal.attentions[c][0, hh, -1, pos] for c, hh in CABEZAS])
    return (sal.logits[0, -1, a] - sal.logits[0, -1, b]).item(), aten.float().reshape(len(CABEZAS), h // 2, w // 2).cpu().numpy()


def crono(f, n: int = 12) -> float:
    t = []
    for _ in range(n + 2):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        f()
        torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t[2:]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--datos", default="out/coco")
    p.add_argument("--n", type=int, default=30, help="imagenes para comprobar la equivalencia")
    a = p.parse_args()
    modelo, proc = cargar(atenciones=True)
    indice = json.loads((Path(a.datos) / "indice.json").read_text())

    # ---- da lo mismo que el forward normal?
    dif_gap, corr, signo = [], [], 0
    for f in indice[:a.n]:
        img = Image.open(COCO / "val2017" / f["imagen"]).convert("RGB")
        textos = [pregunta(f["positiva"]), pregunta(f["negativa"]), GENERICO]
        gaps, aten = Imagen(modelo, proc, img).preguntar(textos)
        for i, t in enumerate(textos):
            g, at = normal(modelo, proc, img, t)
            dif_gap.append(abs(g - gaps[i]))
            signo += (g > 0) == (gaps[i] > 0)
            corr.append(np.corrcoef(at.ravel(), aten[i].ravel())[0, 1])
    print(f"equivalencia en {a.n} imagenes x 3 preguntas:")
    print(f"   |gap normal - gap con prefijo|: media {np.mean(dif_gap):.2f} nats, max {np.max(dif_gap):.2f}; mismo signo {signo}/{len(dif_gap)}")
    print(f"   correlacion de los mapas de las {len(CABEZAS)} cabezas: media {np.mean(corr):.4f}, min {np.min(corr):.4f}")

    # ---- cuanto cuesta
    f = indice[0]
    img = Image.open(COCO / "val2017" / f["imagen"]).convert("RGB")
    nombres = [c["name"] for c in json.loads((COCO / "instances_val2017.json").read_text())["categories"]]
    todas = [pregunta(c) for c in nombres]
    pre = Imagen(modelo, proc, img)
    print(f"\ntiempos (imagen {img.size[0]}x{img.size[1]}, prefijo de {pre.n} tokens):")
    print(f"   forward normal, 1 pregunta                 {crono(lambda: normal(modelo, proc, img, todas[0])):6.0f} ms")
    print(f"   prefijo de la imagen (una vez)             {crono(lambda: Imagen(modelo, proc, img)):6.0f} ms")
    for n in (1, 2, 4, 8, 13):
        ms = crono(lambda: pre.preguntar(todas[:n]))
        print(f"   lote de {n:2d} preguntas sobre el prefijo       {ms:6.0f} ms   ({ms / n:5.1f} ms por pregunta)")
    def ochenta():
        for i in range(0, 81, 9):
            pre.preguntar((todas + [GENERICO])[i:i + 9])
    ms = crono(ochenta, 6)
    print(f"   las 80 clases de COCO + generica (9 lotes) {ms:6.0f} ms")
    print(f"   pico de VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
