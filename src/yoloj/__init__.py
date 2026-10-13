"""YOLOJ: piramide de predicados JEV sobre una imagen, sin entrenar nada."""
from .piramide import Celda, Resultado, piramide
from .readout import Lector, Veredicto

__version__ = "0.0.1"
__all__ = ["Celda", "Lector", "Resultado", "Veredicto", "piramide"]
