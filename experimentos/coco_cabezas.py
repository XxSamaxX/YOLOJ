#!/usr/bin/env python3
"""Elige cabezas de localizacion en la mitad A de COCO y las mide en la mitad B.

Dos medidas por mapa, contra la mascara real del objeto en la rejilla:
  punteria   el maximo del mapa cae en una celda del objeto (pointing game)
  AUC        el mapa ordena las celdas del objeto por delante de las demas

Tres formas de tratar el sumidero (el parche que se lleva la atencion tardia sin
depender de la pregunta):
  cruda      el mapa tal cual
  relativa   dividido por el mapa del predicado generico de esa imagen y cabeza
  tapada     se anulan las K celdas mas atendidas con el predicado generico
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DENTRO = 0.25        # una celda cuenta como objeto si este cubre al menos un cuarto
TAPAR = 3


def auc(valores: np.ndarray, positivos: np.ndarray) -> float:
    """AUC por rangos medios (Mann-Whitney), con empates."""
    v = valores.ravel()
    y = positivos.ravel()
    orden = np.argsort(v, kind="stable")
    rangos = np.empty(len(v))
    rangos[orden] = np.arange(1, len(v) + 1)
    _, inv, cuenta = np.unique(v, return_inverse=True, return_counts=True)
    suma = np.zeros(len(cuenta))
    np.add.at(suma, inv, rangos)
    rangos = (suma / cuenta)[inv]
    n1, n0 = y.sum(), (~y).sum()
    return float((rangos[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def variantes(aten: np.ndarray, gen: np.ndarray) -> dict[str, np.ndarray]:
    """aten, gen: [capas, cabezas, gh, gw] -> lo mismo en cada variante."""
    sumidero = gen[17:35].mean((0, 1))                    # donde va la atencion tardia sin pregunta
    tapa = np.ones_like(sumidero)
    tapa.ravel()[np.argsort(-sumidero, axis=None)[:TAPAR]] = 0
    return {"cruda": aten, "relativa": aten / (gen + 1e-4), "tapada": aten * tapa}


def cargar(d: Path, mitad: str):
    for ficha in json.loads((d / "indice.json").read_text()):
        if ficha["mitad"] != mitad:
            continue
        z = {t: np.load(d / f"{ficha['id']:012d}_{t}.npz") for t in ("pos", "neg", "gen")}
        obj = z["pos"]["mascara"] >= DENTRO
        if obj.any() and not obj.all():
            yield ficha, z, obj


def medir(mapa: np.ndarray, obj: np.ndarray) -> tuple[float, float]:
    return float(obj.ravel()[mapa.argmax()]), auc(mapa, obj)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--datos", default="out/coco")
    a = p.parse_args()
    d = Path(a.datos)

    # ---------------------------------------------------------- mitad A: elegir
    acum: dict[str, list] = {}
    n = 0
    for _, z, obj in cargar(d, "A"):
        for nombre, v in variantes(z["pos"]["aten"].astype(np.float32),
                                   z["gen"]["aten"].astype(np.float32)).items():
            capas, cabezas = v.shape[:2]
            plano = v.reshape(capas, cabezas, -1)
            acierto = obj.ravel()[plano.argmax(-1)].astype(np.float64)
            aucs = np.array([[auc(plano[c, h], obj) for h in range(cabezas)] for c in range(capas)])
            s = acum.setdefault(nombre, [np.zeros_like(acierto), np.zeros_like(aucs)])
            s[0] += acierto
            s[1] += aucs
        n += 1
    print(f"mitad A: {n} imagenes\n")
    elegidas: dict[str, list[tuple[int, int]]] = {}
    for nombre, (acierto, aucs) in acum.items():
        acierto, aucs = acierto / n, aucs / n
        # se ordena por punteria y se desempata por AUC
        orden = np.lexsort((-aucs.ravel(), -acierto.ravel()))[:10]
        elegidas[nombre] = [tuple(int(x) for x in np.unravel_index(i, acierto.shape)) for i in orden]
        print(f"[{nombre}] mejores cabezas en A (punteria / AUC):")
        print("   " + "  ".join(f"{c}.{h} {acierto[c, h]:.0%}/{aucs[c, h]:.2f}" for c, h in elegidas[nombre]))
        por_capa = acierto.max(1)
        print("   mejor punteria por capa: " + " ".join(f"{x:.0%}"[:-1] for x in por_capa) + "\n")

    # ---------------------------------------------------------- mitad B: medir
    filas: dict[str, list[tuple[float, float]]] = {}
    presencia: dict[str, list[tuple[float, float]]] = {}
    gap_capa_pos, gap_capa_neg = [], []
    azar, m = 0.0, 0
    for ficha, z, obj in cargar(d, "B"):
        pos = z["pos"]["aten"].astype(np.float32)
        neg = z["neg"]["aten"].astype(np.float32)
        gen = z["gen"]["aten"].astype(np.float32)
        gh, gw = obj.shape
        azar += obj.mean()
        m += 1
        centro = np.zeros(obj.shape)
        centro[gh // 2, gw // 2] = 1
        yy, xx = np.mgrid[:gh, :gw]
        campana = -((yy - (gh - 1) / 2) ** 2 / gh ** 2 + (xx - (gw - 1) / 2) ** 2 / gw ** 2)
        filas.setdefault("centro de la imagen", []).append(
            (float(obj[gh // 2, gw // 2]), auc(campana, obj)))
        vp, vn = variantes(pos, gen), variantes(neg, gen)
        for nombre in vp:
            filas.setdefault(f"{nombre}: media de todas", []).append(medir(vp[nombre].mean((0, 1)), obj))
            filas.setdefault(f"{nombre}: media capas 17-23", []).append(medir(vp[nombre][17:24].mean((0, 1)), obj))
            for k in (1, 3, 5, 10):
                cab = elegidas[nombre][:k]
                # cada cabeza normalizada a suma 1: cuenta su forma, no cuanta atencion tiene
                suma = sum(vp[nombre][c, h] / (vp[nombre][c, h].sum() + 1e-9) for c, h in cab)
                filas.setdefault(f"{nombre}: {k} elegida{'s' if k > 1 else ''}", []).append(medir(suma, obj))
        # presencia: la masa de atencion a la imagen de las cabezas elegidas, con y sin objeto
        for k in (3, 10):
            cab = elegidas["cruda"][:k]
            presencia.setdefault(f"masa de {k} cabezas (cruda)", []).append(
                (sum(pos[c, h].sum() for c, h in cab), sum(neg[c, h].sum() for c, h in cab)))
            cab = elegidas["tapada"][:k]
            presencia.setdefault(f"masa de {k} cabezas (tapada)", []).append(
                (sum(vp["tapada"][c, h].sum() for c, h in cab), sum(vn["tapada"][c, h].sum() for c, h in cab)))
        presencia.setdefault("gap final", []).append((float(z["pos"]["gap"]), float(z["neg"]["gap"])))
        gap_capa_pos.append(z["pos"]["gap_capa"])
        gap_capa_neg.append(z["neg"]["gap_capa"])

    print(f"mitad B: {m} imagenes; el objeto ocupa de media el {azar / m:.1%} de la rejilla (= punteria al azar)\n")
    print(f"{'mapa':34s} punteria    AUC")
    for nombre, v in filas.items():
        v = np.array(v)
        print(f"{nombre:34s} {v[:, 0].mean():7.1%}  {v[:, 1].mean():6.3f}")

    print("\npresencia (separar la pregunta con objeto de la pregunta sin el, misma imagen):")
    for nombre, v in presencia.items():
        v = np.array(v)
        valores = np.concatenate([v[:, 0], v[:, 1]])
        etiquetas = np.concatenate([np.ones(len(v), bool), np.zeros(len(v), bool)])
        print(f"   {nombre:30s} AUC {auc(valores, etiquetas):.3f}   medias {v[:, 0].mean():+.2f} / {v[:, 1].mean():+.2f}")

    gp, gn = np.array(gap_capa_pos), np.array(gap_capa_neg)       # [m, capas+1]
    print("\ngap leido en cada capa (lente): acierto de signo y AUC si / no")
    for c in range(1, gp.shape[1]):
        acierto = ((gp[:, c] > 0).mean() + (gn[:, c] < 0).mean()) / 2
        valores = np.concatenate([gp[:, c], gn[:, c]])
        etiquetas = np.concatenate([np.ones(len(gp), bool), np.zeros(len(gn), bool)])
        print(f"   capa {c - 1:2d}: signo {acierto:6.1%}   AUC {auc(valores, etiquetas):.3f}")


if __name__ == "__main__":
    main()
