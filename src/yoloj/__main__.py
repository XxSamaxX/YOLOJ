"""python -m yoloj foto.jpg "is there a cat?" """
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .piramide import piramide
from .readout import Lector
from .viz import panel, texto


def main() -> None:
    p = argparse.ArgumentParser(prog="yoloj", description=__doc__)
    p.add_argument("imagen")
    p.add_argument("pregunta", help="predicado si/no, mejor en ingles")
    p.add_argument("--niveles", default="1,2,4", help="celdas por lado en cada nivel")
    p.add_argument("--poda", type=float, default=-3.0, help="gap por debajo del cual no se baja")
    p.add_argument("--sin-poda", action="store_true", help="evalua todas las celdas")
    p.add_argument("--solape", type=float, default=0.0, help="fraccion de celda hacia cada vecina")
    p.add_argument("--lado-min", type=int, default=224)
    p.add_argument("--calibrar", choices=("auto", "siempre", "nunca"), default="auto")
    p.add_argument("--url", help="endpoint /v1 de llama-server (o YOLOJ_URL)")
    p.add_argument("--modelo", help="nombre del modelo (o YOLOJ_MODEL)")
    p.add_argument("--salida", default="out", help="carpeta para el JSON y el PNG")
    a = p.parse_args()

    r = piramide(
        a.imagen, a.pregunta, Lector(a.url, a.modelo),
        niveles=tuple(int(n) for n in a.niveles.split(",")),
        poda=None if a.sin_poda else a.poda,
        solape=a.solape, lado_min=a.lado_min, calibrar=a.calibrar,
    )
    print(texto(r))
    print(f"{r.evaluadas}/{len(r.celdas)} celdas evaluadas, "
          f"{r.forwards} forwards, {r.segundos:.1f} s")

    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    nombre = f"{Path(a.imagen).stem}__{re.sub(r'[^a-z0-9]+', '-', a.pregunta.lower()).strip('-')[:40]}"
    (salida / f"{nombre}.json").write_text(json.dumps(r.como_dict(), indent=1, ensure_ascii=False))
    panel(a.imagen, r).save(salida / f"{nombre}.png")
    print(salida / f"{nombre}.png")


if __name__ == "__main__":
    main()
