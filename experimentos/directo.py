#!/usr/bin/env python3
"""En directo: se escribe una pregunta y un punto marca la respuesta sobre el video.

    directo.py                          la Kinect (/dev/video10)
    directo.py --fuente video.mp4       un fichero

Abre http://127.0.0.1:8765 en el navegador. En la caja se escribe un nombre
("bottle"), una pregunta completa ("Is anyone wearing glasses?") o varias
separadas por punto y coma. Por cada fotograma que le da tiempo a mirar, el 4B
contesta si/no y, si es que si, el pico del mapa relativo de sus cabezas de
localizacion pone el punto. No hay SAM: solo el 4B.

El video se ve fluido; el punto es el del ultimo fotograma analizado.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from coco_atencion import pregunta
from desgrane import GENERICO, cargar
from foto import mapa_relativo
from prefijo import Imagen

COLORES = [(255, 60, 60), (60, 220, 90), (70, 140, 255), (255, 200, 40), (230, 90, 230), (60, 220, 220)]
PAGINA = """<!doctype html><meta charset=utf-8><title>YOLOJ en directo</title>
<style>body{background:#111;color:#ddd;font:16px sans-serif;margin:16px;text-align:center}
img{max-width:100%;border:1px solid #333}input{width:min(720px,80%);font-size:18px;padding:8px;margin:12px 0}
#e{font-family:monospace;white-space:pre;color:#9c9}</style>
<img id=v><br>
<input id=q autofocus placeholder="bottle; Is anyone wearing glasses?  (Intro para enviar)">
<div id=e></div>
<script>
q.onkeydown=e=>{if(e.key=="Enter")fetch("/pregunta",{method:"POST",body:q.value})}
setInterval(async()=>{try{e.textContent=await(await fetch("/estado")).text()}catch(x){e.textContent="sin conexion con directo.py"}},500)
// fotograma a fotograma en vez de un flujo continuo: si una peticion falla, la siguiente lo recupera
;(async()=>{for(;;){try{const r=await fetch("/foto?"+Date.now());if(r.ok){const u=URL.createObjectURL(await r.blob());
await new Promise(ok=>{v.onload=v.onerror=ok;v.src=u});URL.revokeObjectURL(u)}}catch(x){}
await new Promise(ok=>setTimeout(ok,50))}})()
</script>"""


class Estado:
    def __init__(self):
        self.candado = threading.Lock()
        self.fotograma: Image.Image | None = None
        self.n = 0                      # cuantos fotogramas han llegado
        self.t_fotograma = 0.0          # cuando llego el ultimo
        self.preguntas: list[str] = []
        self.puntos: list[dict] = []    # del ultimo fotograma analizado
        self.ms = {}
        self.error = ""


def leer_video(est: Estado, fuente: str, ancho: int, alto: int) -> None:
    entrada = ["-f", "v4l2", "-i", fuente] if fuente.startswith("/dev/") else ["-re", "-stream_loop", "-1", "-i", fuente]
    orden = ["ffmpeg", "-loglevel", "error", *entrada, "-vf", f"scale={ancho}:{alto}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(orden, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tam = ancho * alto * 3
    while True:
        datos = p.stdout.read(tam)
        if len(datos) < tam:
            est.error = "se corto el video: " + p.stderr.read().decode(errors="replace")[-300:]
            return
        img = Image.frombuffer("RGB", (ancho, alto), datos, "raw", "RGB", 0, 1)
        with est.candado:
            est.fotograma, est.n, est.t_fotograma = img, est.n + 1, time.monotonic()


def pico(m: np.ndarray) -> tuple[float, float]:
    """El maximo del mapa, afinado con el centro de masas de su vecindad 3x3. Devuelve (x, y) en [0, 1]."""
    y, x = np.unravel_index(m.argmax(), m.shape)
    y0, y1, x0, x1 = max(0, y - 1), min(m.shape[0], y + 2), max(0, x - 1), min(m.shape[1], x + 2)
    v = m[y0:y1, x0:x1]
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return float(((xs + 0.5) * v).sum() / v.sum() / m.shape[1]), float(((ys + 0.5) * v).sum() / v.sum() / m.shape[0])


@torch.inference_mode()
def mirar(est: Estado, modelo, proc, lado: int, corte: float) -> None:
    visto = 0
    while True:
        with est.candado:
            img, n, preguntas = est.fotograma, est.n, list(est.preguntas)
        if img is None or n == visto or not preguntas:
            if not preguntas:
                est.puntos = []
            time.sleep(0.01)
            continue
        visto = n
        try:
            t0 = time.perf_counter()
            f = min(1.0, lado / max(img.size))
            pre = Imagen(modelo, proc, img.resize((round(img.width * f), round(img.height * f)), Image.BILINEAR))
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            textos = [q if q.endswith("?") else pregunta(q) for q in preguntas]
            gaps, atens = pre.preguntar(textos + [GENERICO])
            puntos = []
            for i, q in enumerate(preguntas):
                d = {"pregunta": q, "gap": float(gaps[i]), "color": COLORES[i % len(COLORES)]}
                if gaps[i] > corte:
                    d["x"], d["y"] = pico(mapa_relativo(atens[i], atens[-1]))
                puntos.append(d)
            t2 = time.perf_counter()
            est.puntos = puntos
            est.ms = {"prefijo": (t1 - t0) * 1000, "preguntas": (t2 - t1) * 1000, "rejilla": pre.rejilla}
            est.error = ""
            time.sleep(0.03)                        # un respiro: el escritorio pinta con la misma grafica
        except Exception as e:                      # una pregunta rara no debe tumbar el directo
            est.error = f"{type(e).__name__}: {e}"[:300]
            time.sleep(0.5)


def pintar(img: Image.Image, puntos: list[dict], fuente) -> bytes:
    img = img.copy()
    d = ImageDraw.Draw(img)
    r = max(6, img.width // 110)
    for k, p in enumerate(puntos):
        if "x" in p:
            x, y = p["x"] * img.width, p["y"] * img.height
            d.ellipse([x - r, y - r, x + r, y + r], fill=p["color"], outline=(255, 255, 255), width=3)
            d.text((x + r + 5, y), p["pregunta"], font=fuente, anchor="lm", fill=(255, 255, 255),
                   stroke_width=3, stroke_fill=(0, 0, 0))
        marca = "si" if "x" in p else "no"
        d.text((10, 10 + k * (fuente.size + 6)), f"{p['pregunta']}: {marca} ({p['gap']:+.0f})", font=fuente,
               fill=p["color"] if "x" in p else (170, 170, 170), stroke_width=3, stroke_fill=(0, 0, 0))
    b = io.BytesIO()
    img.save(b, "JPEG", quality=80)
    return b.getvalue()


def servidor(est: Estado, puerto: int) -> ThreadingHTTPServer:
    fuente = ImageFont.load_default(size=20)

    def jpeg() -> bytes | None:
        with est.candado:
            img = est.fotograma
        return pintar(img, est.puntos, fuente) if img is not None else None

    class Peticion(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _enviar(self, tipo: str, cuerpo: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)

        def do_GET(self):
            if self.path == "/":
                self._enviar("text/html", PAGINA.encode())
            elif self.path == "/estado":
                ms = est.ms
                lineas = [est.error] if est.error else []
                if time.monotonic() - est.t_fotograma > 2:
                    lineas.append("sin senal de la camara desde hace %.0f s" % (time.monotonic() - est.t_fotograma)
                                  if est.n else "esperando a la camara")
                if ms and est.puntos:
                    total = ms["prefijo"] + ms["preguntas"]
                    lineas.append(f"{1000 / total:.1f} analisis por segundo   (imagen {ms['prefijo']:.0f} ms + preguntas "
                                  f"{ms['preguntas']:.0f} ms, rejilla {ms['rejilla'][0]}x{ms['rejilla'][1]})")
                self._enviar("text/plain; charset=utf-8", "\n".join(lineas).encode())
            elif self.path.startswith("/foto"):
                j = jpeg()
                self._enviar("image/jpeg", j) if j else self.send_error(503)
            elif self.path == "/video":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                visto = -1
                try:
                    while True:
                        if est.n == visto:
                            time.sleep(0.005)
                            continue
                        visto = est.n
                        j = jpeg()
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j) + j + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            texto = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode(errors="replace")
            with est.candado:
                est.preguntas = [q.strip() for q in texto.split(";") if q.strip()][:6]
            self._enviar("text/plain", json.dumps(est.preguntas).encode())

    return ThreadingHTTPServer(("127.0.0.1", puerto), Peticion)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--fuente", default="/dev/video10")
    p.add_argument("--ancho", type=int, default=960)
    p.add_argument("--alto", type=int, default=540)
    p.add_argument("--lado", type=int, default=640, help="lado largo de lo que ve el 4B; mas = mas fino y mas lento")
    p.add_argument("--corte", type=float, default=0.0, help="gap minimo para dar la respuesta por un si")
    p.add_argument("--puerto", type=int, default=8765)
    a = p.parse_args()

    est = Estado()
    modelo, proc = cargar(atenciones=True)
    threading.Thread(target=leer_video, args=(est, a.fuente, a.ancho, a.alto), daemon=True).start()
    threading.Thread(target=mirar, args=(est, modelo, proc, a.lado, a.corte), daemon=True).start()
    print(f"http://127.0.0.1:{a.puerto}", flush=True)
    servidor(est, a.puerto).serve_forever()


if __name__ == "__main__":
    main()
