"""Readout JEV contra un llama-server: la respuesta no se genera, se lee.

Un predicado si/no sobre una imagen cuesta un prefill y un token. Del primer
token solo interesan dos logprobs, los de las letras de opcion; su diferencia es
el gap en nats, que es la misma que la diferencia de logits.

Frente a discern (transformers, logits completos) cambian dos cosas:
  1. el servidor devuelve los top-N logprobs, no el vocabulario entero. Con
     N=1000 las dos letras han salido siempre; si una faltase, se acota por
     abajo con el minimo devuelto y el veredicto lo dice (`acotado`).
  2. hay que apagar el razonamiento del modelo (`enable_thinking: false`), o el
     primer token es el del pensamiento y no la letra.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import asdict, dataclass, field

import requests
from PIL import Image

# El mismo contrato de prompt que SemIf (Theo C. J. Lee, MIT) y discern, para
# que los gaps sean comparables con los ya medidos.
SISTEMA = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
LETRAS = "AB"
URL_POR_DEFECTO = "http://127.0.0.1:8080/v1"   # el puerto por defecto de llama-server


@dataclass
class Veredicto:
    gap: float                      # nats; positivo = si, negativo = no
    forwards: int
    por_orden: list[float]          # gap de cada orden de opciones, sin promediar
    tokens: int                     # tokens de prompt del ultimo forward
    segundos: float
    acotado: bool = False           # alguna letra no estaba en el top-N
    extra: dict = field(default_factory=dict)

    @property
    def si(self) -> bool:
        return self.gap > 0

    def como_dict(self) -> dict:
        return asdict(self)


def _a_data_url(imagen: Image.Image) -> str:
    # PNG: los recortes son pequenos y asi no se anade ruido de JPEG al predicado
    buf = io.BytesIO()
    imagen.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Lector:
    """Cliente de un endpoint OpenAI-compatible de llama-server con vision."""

    def __init__(self, url: str | None = None, modelo: str | None = None,
                 top: int = 1000, timeout: float = 300.0):
        self.url = (url or os.environ.get("YOLOJ_URL") or URL_POR_DEFECTO).rstrip("/")
        self.modelo = modelo or os.environ.get("YOLOJ_MODEL") or "default"
        self.top = top
        self.timeout = timeout
        self.sesion = requests.Session()

    def _forward(self, data_url: str, pregunta: str, si_primero: bool) -> tuple[float, int, bool]:
        """Un forward. Devuelve (gap si-no, tokens de prompt, acotado)."""
        opciones = ("yes", "no") if si_primero else ("no", "yes")
        carga = {
            "criterion": pregunta,
            "options": [{"letter": LETRAS[i], "description": o} for i, o in enumerate(opciones)],
        }
        cuerpo = {
            "model": self.modelo,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": self.top,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": SISTEMA},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": json.dumps(carga, ensure_ascii=False)},
                ]},
            ],
        }
        r = self.sesion.post(self.url + "/chat/completions", json=cuerpo, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        try:
            top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                "El servidor no devolvio logprobs del primer token. "
                "Hace falta un llama-server reciente con --mmproj."
            ) from error
        lp = {t["token"]: t["logprob"] for t in top}
        suelo = min(lp.values())
        acotado = any(letra not in lp for letra in LETRAS)
        a, b = lp.get("A", suelo), lp.get("B", suelo)
        gap = (a - b) if si_primero else (b - a)
        return gap, int(d.get("usage", {}).get("prompt_tokens", 0)), acotado

    def predicado(self, imagen: Image.Image, pregunta: str,
                  calibrar: str = "auto", duda: float = 6.0) -> Veredicto:
        """Gap en nats de un predicado si/no sobre una imagen.

        calibrar:
          "siempre"  dos forwards, uno por orden de opciones, y se promedia.
          "nunca"    un forward con "yes" en la letra A.
          "auto"     el segundo forward solo si el primero queda por debajo de
                     `duda` nats en valor absoluto.

        El sesgo hacia la letra A medido en Qwen3.8-27B a 2 bits es de ~2,4 nats
        (gato en el sofa: +10,4 con yes=A, +5,6 con yes=B). Por encima de 6 el
        signo no puede cambiar al promediar; por debajo si.
        """
        if calibrar not in ("auto", "siempre", "nunca"):
            raise ValueError("calibrar debe ser 'auto', 'siempre' o 'nunca'")
        data_url = _a_data_url(imagen)
        t0 = time.perf_counter()
        g, tokens, acotado = self._forward(data_url, pregunta, si_primero=True)
        gaps = [g]
        if calibrar == "siempre" or (calibrar == "auto" and abs(g) < duda):
            g2, tokens, acotado2 = self._forward(data_url, pregunta, si_primero=False)
            gaps.append(g2)
            acotado = acotado or acotado2
        return Veredicto(
            gap=sum(gaps) / len(gaps),
            forwards=len(gaps),
            por_orden=gaps,
            tokens=tokens,
            segundos=time.perf_counter() - t0,
            acotado=acotado,
        )
