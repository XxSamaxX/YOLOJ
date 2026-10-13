"""Piramide de predicados: lo que hace un FPN, sin cabezas entrenadas.

La imagen se mira a varias escalas (entera, rejilla 2x2, 4x4...) y en cada celda
se lee el gap de un predicado si/no. Se recorre de arriba abajo: una celda gruesa
con un "no" claro poda a sus hijas, y solo se baja donde hay senal o duda.

La poda supone que el predicado es de existencia ("hay un gato"): si esta en una
celda hija, tambien esta en la madre. Lo que la madre no ve por pequeno se pierde.
Para medir cuanto se pierde, `poda=None` evalua todas las celdas.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from PIL import Image

from .readout import Lector


@dataclass
class Celda:
    nivel: int                      # indice en `niveles`
    rejilla: int                    # celdas por lado en ese nivel
    fila: int
    col: int
    caja: tuple[int, int, int, int]  # x0, y0, x1, y1 del recorte enviado, en pixeles
    estado: str                     # "evaluada" | "podada"
    gap: float                      # propio si evaluada; el del ancestro si podada
    forwards: int = 0
    tokens: int = 0
    segundos: float = 0.0
    acotado: bool = False


@dataclass
class Resultado:
    pregunta: str
    tamano: tuple[int, int]
    niveles: tuple[int, ...]
    poda: float | None
    celdas: list[Celda]
    segundos: float

    def nivel(self, i: int) -> list[list[Celda]]:
        """Las celdas del nivel i como matriz fila x columna."""
        n = self.niveles[i]
        matriz: list[list[Celda | None]] = [[None] * n for _ in range(n)]
        for c in self.celdas:
            if c.nivel == i:
                matriz[c.fila][c.col] = c
        return matriz  # type: ignore[return-value]

    def mapa(self) -> list[list[float]]:
        """Gaps del nivel mas fino: la localizacion aproximada del predicado."""
        return [[c.gap for c in fila] for fila in self.nivel(len(self.niveles) - 1)]

    @property
    def forwards(self) -> int:
        return sum(c.forwards for c in self.celdas)

    @property
    def evaluadas(self) -> int:
        return sum(c.estado == "evaluada" for c in self.celdas)

    def como_dict(self) -> dict:
        return {
            "pregunta": self.pregunta,
            "tamano": list(self.tamano),
            "niveles": list(self.niveles),
            "poda": self.poda,
            "segundos": self.segundos,
            "forwards": self.forwards,
            "evaluadas": self.evaluadas,
            "celdas_totales": len(self.celdas),
            "celdas": [asdict(c) for c in self.celdas],
        }


def _caja(ancho: int, alto: int, n: int, fila: int, col: int, solape: float) -> tuple[int, int, int, int]:
    cw, ch = ancho / n, alto / n
    x0, y0 = (col - solape) * cw, (fila - solape) * ch
    x1, y1 = (col + 1 + solape) * cw, (fila + 1 + solape) * ch
    return (max(0, round(x0)), max(0, round(y0)), min(ancho, round(x1)), min(alto, round(y1)))


def _recorte(imagen: Image.Image, caja: tuple[int, int, int, int], lado_min: int) -> Image.Image:
    trozo = imagen.crop(caja)
    corto = min(trozo.size)
    if corto < lado_min:
        f = lado_min / corto
        trozo = trozo.resize((round(trozo.width * f), round(trozo.height * f)), Image.BICUBIC)
    return trozo


def piramide(imagen: Image.Image | str, pregunta: str, lector: Lector | None = None,
             niveles: tuple[int, ...] = (1, 2, 4), poda: float | None = -3.0,
             solape: float = 0.0, lado_min: int = 224,
             calibrar: str = "auto", duda: float = 6.0) -> Resultado:
    """Evalua `pregunta` en cada celda de cada nivel, podando de arriba abajo.

    niveles   celdas por lado en cada nivel; cada uno multiplo del anterior, para
              que toda celda tenga una unica madre.
    poda      una celda con gap por debajo de este valor no baja: sus hijas
              heredan su gap sin gastar forwards. None = sin poda.
    solape    fraccion de celda que el recorte se extiende hacia cada vecina,
              para no partir por la mitad lo que cae en una frontera.
    lado_min  los recortes con el lado corto menor se amplian hasta aqui.
    """
    if isinstance(imagen, str):
        imagen = Image.open(imagen)
    imagen = imagen.convert("RGB")
    niveles = tuple(niveles)
    for anterior, actual in zip(niveles, niveles[1:]):
        if actual <= anterior or actual % anterior:
            raise ValueError("cada nivel debe ser un multiplo mayor del anterior, p. ej. 1,2,4")
    lector = lector or Lector()
    ancho, alto = imagen.size
    t0 = time.perf_counter()

    celdas: list[Celda] = []
    madres: dict[tuple[int, int], Celda] = {}
    for i, n in enumerate(niveles):
        actuales: dict[tuple[int, int], Celda] = {}
        for fila in range(n):
            for col in range(n):
                caja = _caja(ancho, alto, n, fila, col, solape if n > 1 else 0.0)
                madre = None
                if i:
                    k = n // niveles[i - 1]
                    madre = madres[(fila // k, col // k)]
                if madre is not None and poda is not None and (
                        madre.estado == "podada" or madre.gap < poda):
                    celda = Celda(i, n, fila, col, caja, "podada", madre.gap)
                else:
                    v = lector.predicado(_recorte(imagen, caja, lado_min), pregunta,
                                         calibrar=calibrar, duda=duda)
                    celda = Celda(i, n, fila, col, caja, "evaluada", v.gap,
                                  v.forwards, v.tokens, v.segundos, v.acotado)
                actuales[(fila, col)] = celda
                celdas.append(celda)
        madres = actuales

    return Resultado(pregunta, (ancho, alto), niveles, poda, celdas, time.perf_counter() - t0)
