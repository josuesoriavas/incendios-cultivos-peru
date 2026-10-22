"""
Paso 3: exporta el raster de Curahuasi para el visor web (una fila de píxeles por registro, en base64).

raster_curahuasi.json (20 m): por fila, tres capas de un byte por píxel
  dnbr  = floor((dNBR + 0.5) * 100), entre 1 y 255; 0 = sin dato (nube, sombra o fuera de la imagen)
  flags = bit 1 zona a <= 1 km de focos · bit 2 agua · bit 4 superficie agrícola SAN 2024
  wc    = clase ESA WorldCover 2021
color_curahuasi.json (40 m): color real antes y después (R, G, B intercalados), estirado 0-255.
Uso:  uv run python src/03_raster_web.py
"""
import base64
import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
z = np.load(RAIZ / "data/processed/curahuasi/capas_curahuasi.npz")
OUT = RAIZ / "data/monitor"
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype="uint8").tobytes()).decode()

sin = z["sin_dato"] | ~np.isfinite(z["dnbr"])
q = np.where(sin, 0, np.clip(np.floor((np.nan_to_num(z["dnbr"]) + 0.5) * 100 + 1e-6), 1, 255)).astype("uint8")
flags = (z["zona"] * 1 | z["agua"] * 2 | z["san"] * 4).astype("uint8")
H, W = q.shape
tr = z["transform"]
filas = [{"fila": i, "dnbr": b64(q[i]), "flags": b64(flags[i]), "wc": b64(z["wc"][i])} for i in range(H)]
meta = {"ancho": W, "alto": H, "pixel_m": float(tr[0]), "x0": float(tr[2]), "y0": float(tr[5]), "crs": str(z["crs"])}
(OUT / "raster_curahuasi.json").write_text(json.dumps({"rows": [dict(f, **meta) if i == 0 else f for i, f in enumerate(filas)]}), encoding="utf-8")


def color(rgb):
    a = rgb[:, :H // 2 * 2, :W // 2 * 2].reshape(3, H // 2, 2, W // 2, 2).mean(axis=(2, 4))  # 40 m
    a = np.clip(np.nan_to_num(a) / 0.25, 0, 1) ** 0.8 * 255  # estiramiento simple para la sierra
    return np.moveaxis(a, 0, -1).astype("uint8")  # (fila, columna, RGB)


ca, cd = color(z["rgb_antes"]), color(z["rgb_despues"])
(OUT / "color_curahuasi.json").write_text(json.dumps({"rows": [{"fila": i, "antes": b64(ca[i]), "despues": b64(cd[i])} for i in range(ca.shape[0])]}), encoding="utf-8")
for f in ["raster_curahuasi.json", "color_curahuasi.json"]:
    print(f, round((OUT / f).stat().st_size / 1e6, 2), "MB")
print(meta)
