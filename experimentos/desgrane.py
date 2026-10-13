#!/usr/bin/env python3
"""Desgrane de Qwen3-VL-4B: que sabe cada capa de un predicado, y donde mira.

Un solo forward por (imagen, predicado), con atenciones y estados ocultos. De ahi:

  gap final      logit(si) - logit(no) en la ultima posicion: el readout JEV.
  gap por capa   el mismo readout aplicado al estado de cada capa (logit lens):
                 en que capa queda decidida la respuesta.
  atencion       de la posicion de respuesta a cada token de imagen, por capa y
                 cabeza. Los tokens de imagen son una rejilla (32x32 px cada
                 uno), asi que esto ya es un mapa.
  lente          logit de una palabra ("cat") leido en cada token de imagen, por
                 capa: que parches "dicen" esa palabra.

Guarda todo en un .npz por caso y pinta paneles. No decide nada: es para mirar.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image, ImageDraw, ImageFont

MODELO = "Qwen/Qwen3-VL-4B-Instruct"
SISTEMA = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
# Predicado sin contenido: su atencion es el "fondo" que se resta al normalizar,
# como la atencion relativa de ViCrop (Zhang et al., ICLR 2025).
GENERICO = "Is this an image?"


def cargar(atenciones: bool):
    # eager es la unica implementacion que devuelve los pesos de atencion
    modelo = transformers.AutoModelForImageTextToText.from_pretrained(
        MODELO, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="eager" if atenciones else "sdpa",
    ).eval()
    return modelo, transformers.AutoProcessor.from_pretrained(MODELO)


def entradas(proc, imagen: Image.Image, pregunta: str, si_primero: bool = True):
    opciones = ("yes", "no") if si_primero else ("no", "yes")
    carga = {"criterion": pregunta,
             "options": [{"letter": "AB"[i], "description": o} for i, o in enumerate(opciones)]}
    mensajes = [
        {"role": "system", "content": [{"type": "text", "text": SISTEMA}]},
        {"role": "user", "content": [{"type": "image", "image": imagen},
                                     {"type": "text", "text": json.dumps(carga)}]},
    ]
    return proc.apply_chat_template(mensajes, tokenize=True, add_generation_prompt=True,
                                    return_dict=True, return_tensors="pt")


def _un_token(tok, texto: str) -> int:
    ids = tok.encode(texto, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{texto!r} no es un solo token: {ids}")
    return ids[0]


@torch.inference_mode()
def desgranar(modelo, proc, imagen: Image.Image, pregunta: str, palabra: str | None) -> dict:
    tok = proc.tokenizer
    ent = entradas(proc, imagen, pregunta).to(modelo.device)
    ids = ent["input_ids"][0]
    pos_img = (ids == modelo.config.image_token_id).nonzero().squeeze(1)
    _, h, w = ent["image_grid_thw"][0].tolist()
    m = modelo.config.vision_config.spatial_merge_size
    gh, gw = h // m, w // m
    assert gh * gw == len(pos_img), (gh, gw, len(pos_img))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    sal = modelo(**ent, output_attentions=True, output_hidden_states=True,
                 use_cache=False, return_dict=True)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000

    a, b = _un_token(tok, "A"), _un_token(tok, "B")
    gap = (sal.logits[0, -1, a] - sal.logits[0, -1, b]).item()

    # atencion de la posicion de respuesta a los tokens de imagen: [capas, cabezas, gh, gw]
    aten = torch.stack([c[0, :, -1, :] for c in sal.attentions]).float()
    masa_img = aten[:, :, pos_img].sum(-1)                  # cuanta atencion va a la imagen
    aten_img = aten[:, :, pos_img].reshape(len(sal.attentions), -1, gh, gw)

    # logit lens: norma final + cabeza de vocabulario sobre el estado de cada capa
    lm = modelo.model.language_model
    cabeza = modelo.lm_head
    ocultos = torch.stack(sal.hidden_states)[:, 0]          # [capas+1, seq, d]
    ultimo = cabeza(lm.norm(ocultos[:, -1]))
    gap_capa = (ultimo[:, a] - ultimo[:, b]).float()

    lente = None
    if palabra:
        # la palabra con y sin espacio delante; cuenta la que mas suene en cada parche
        cand = {t for t in (tok.encode(p, add_special_tokens=False)[0]
                            for p in (palabra, " " + palabra, palabra.capitalize()))}
        # capa a capa: el vocabulario entero de todas las capas a la vez no cabe en 12 GB
        prob = torch.stack([
            cabeza(lm.norm(capa[pos_img])).float().log_softmax(-1)[:, sorted(cand)].amax(-1)
            for capa in ocultos])
        lente = prob.reshape(-1, gh, gw).cpu().numpy()

    return {
        "gap": gap, "ms": ms, "tokens": len(ids), "rejilla": (gh, gw),
        "gap_capa": gap_capa.cpu().numpy(),
        "masa_img": masa_img.cpu().numpy(),                 # [capas, cabezas]
        "aten": aten_img.cpu().numpy(),                     # [capas, cabezas, gh, gw]
        "lente": lente,                                     # [capas+1, gh, gw] o None
    }


@torch.inference_mode()
def cronometro(modelo, proc, imagen: Image.Image, pregunta: str, n: int = 8) -> float:
    """ms por forward tal como iria en produccion: sin atenciones ni estados."""
    ent = entradas(proc, imagen, pregunta).to(modelo.device)
    tiempos = []
    for _ in range(n + 2):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        modelo(**ent, use_cache=False, return_dict=True)
        torch.cuda.synchronize()
        tiempos.append((time.perf_counter() - t0) * 1000)
    return float(np.median(tiempos[2:]))


# ---------------------------------------------------------------- dibujo
def _fuente(px: int):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def _calor(base: Image.Image, mapa: np.ndarray, titulo: str, ancho: int = 300) -> Image.Image:
    """Mapa normalizado a [0, 1] por su propio rango, sobre la foto en gris."""
    alto = round(base.height * ancho / base.width)
    gris = base.convert("L").convert("RGB").resize((ancho, alto))
    v = mapa.astype(np.float64)
    v = (v - v.min()) / (v.max() - v.min() + 1e-12)
    capa = Image.fromarray((v * 255).astype(np.uint8)).resize((ancho, alto), Image.BICUBIC)
    rojo = Image.merge("RGB", (capa, Image.new("L", capa.size, 40), Image.new("L", capa.size, 40)))
    mezcla = Image.composite(rojo, gris, capa.point(lambda x: int(x * 0.85)))
    lienzo = Image.new("RGB", (ancho, alto + 20), (18, 18, 18))
    lienzo.paste(mezcla, (0, 20))
    ImageDraw.Draw(lienzo).text((4, 3), titulo, font=_fuente(13), fill=(235, 235, 235))
    return lienzo


def _mosaico(celdas: list[Image.Image], columnas: int) -> Image.Image:
    w, h = celdas[0].size
    filas = -(-len(celdas) // columnas)
    lienzo = Image.new("RGB", (columnas * (w + 4) + 4, filas * (h + 4) + 4), (18, 18, 18))
    for i, c in enumerate(celdas):
        lienzo.paste(c, (4 + (i % columnas) * (w + 4), 4 + (i // columnas) * (h + 4)))
    return lienzo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("casos", help="JSON: lista de {imagen, pregunta, palabra}")
    p.add_argument("--salida", default="out/desgrane")
    a = p.parse_args()
    casos = json.loads(Path(a.casos).read_text())
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)

    modelo, proc = cargar(atenciones=True)
    print(f"VRAM tras cargar: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    fondo: dict[str, np.ndarray] = {}

    for caso in casos:
        img = Image.open(caso["imagen"]).convert("RGB")
        nombre = f"{Path(caso['imagen']).stem}__{re.sub(r'[^a-z0-9]+', '-', caso['pregunta'].lower()).strip('-')[:36]}"
        r = desgranar(modelo, proc, img, caso["pregunta"], caso.get("palabra"))
        if caso["imagen"] not in fondo:
            fondo[caso["imagen"]] = desgranar(modelo, proc, img, GENERICO, None)["aten"]
        capas = r["aten"].shape[0]
        media = r["aten"].mean(1)                                # [capas, gh, gw]
        relativa = media / (fondo[caso["imagen"]].mean(1) + 1e-6)

        print(f"\n=== {nombre}\n    gap {r['gap']:+.1f} nats | {r['tokens']} tokens "
              f"| rejilla {r['rejilla'][0]}x{r['rejilla'][1]} | {r['ms']:.0f} ms con atenciones")
        print("    capa  gap(lente)  atencion a la imagen (media / cabeza max)")
        for c in range(capas):
            print(f"    {c:4d}  {r['gap_capa'][c + 1]:+9.1f}   "
                  f"{r['masa_img'][c].mean():.3f} / {r['masa_img'][c].max():.3f}")

        np.savez_compressed(salida / f"{nombre}.npz",
                            **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})
        _mosaico([_calor(img, media[c], f"capa {c}  atencion") for c in range(capas)], 6
                 ).save(salida / f"{nombre}__atencion.png")
        _mosaico([_calor(img, relativa[c], f"capa {c}  relativa") for c in range(capas)], 6
                 ).save(salida / f"{nombre}__relativa.png")
        if r["lente"] is not None:
            _mosaico([_calor(img, r["lente"][c + 1], f"capa {c}  '{caso['palabra']}'")
                      for c in range(capas)], 6).save(salida / f"{nombre}__lente.png")

    # el coste real: sin atenciones, con la atencion rapida
    del modelo
    torch.cuda.empty_cache()
    modelo, proc = cargar(atenciones=False)
    img = Image.open(casos[0]["imagen"]).convert("RGB")
    print(f"\nforward sin atenciones (sdpa): {cronometro(modelo, proc, img, casos[0]['pregunta']):.0f} ms")


if __name__ == "__main__":
    main()
