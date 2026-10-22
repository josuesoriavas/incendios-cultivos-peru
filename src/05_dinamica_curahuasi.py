"""
Paso 5: dinámica agrícola de Curahuasi (permanente / transitorio / barbecho) con la regla de la app Dinámica Agrícola
del MIDAGRI, la misma del cuaderno 17 del proyecto de clasificación de cultivos.

Serie de los 12 meses previos al incendio (oct 2025 a sep 2026), Sentinel-2 L2A del tile 18LYK, NDVI = (B08 - B04)/(B08 + B04)
en la grilla de 20 m del dNBR; nubes, sombras y cirros enmascarados con la SCL; por mes, el NDVI máximo.
Regla: barbecho si el máximo de los 12 meses <= 0,3; permanente si el mínimo > 0,3; transitorio en el resto;
sin dato si hay menos de 3 meses válidos.
Salidas: data/processed/curahuasi/dinamica_curahuasi.npz y dinamica_resumen.json
Uso:  uv run python src/05_dinamica_curahuasi.py
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt

RAIZ = Path(__file__).resolve().parent.parent
SAL = RAIZ / "data/processed/curahuasi"
BBOX = (-72.57671, -13.72427, -72.3556, -13.50937)  # el mismo recuadro del dNBR
TILE = "MGRS-18LYK"  # el mismo tile: la grilla coincide píxel a píxel
SCL_MALOS = [0, 1, 3, 8, 9, 10]
CLASES = ["sin dato", "barbecho", "transitorio", "permanente"]
z = np.load(SAL / "capas_curahuasi.npz")
H, W = z["dnbr"].shape
TR = tuple(z["transform"])


def leer(href, forma=None):
    for intento in range(4):
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2"), rasterio.open(href) as src:
                b = transform_bounds(4326, src.crs, *BBOX)
                w = from_bounds(*b, src.transform).round_offsets().round_lengths()
                a = src.read(1, window=w, out_shape=forma, resampling=Resampling.bilinear if forma else Resampling.nearest)
                return a, src.window_transform(w)
        except Exception as ex:  # errores transitorios de red
            print(f"    reintento {intento + 1}: {ex}", flush=True)
            time.sleep(3)
    raise RuntimeError(href)


def ndvi(it):
    scl, tr = leer(it.assets["scl"].href)
    if scl.shape != (H, W) or not np.allclose(tuple(tr)[:6], TR):
        raise ValueError(f"{it.id}: la grilla no coincide con la del dNBR")
    red = leer(it.assets["red"].href, (H, W))[0].astype("float32")
    nir = leer(it.assets["nir"].href, (H, W))[0].astype("float32")
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (nir - red) / (nir + red)
    v[np.isin(scl, SCL_MALOS) | (nir == 0) | (red == 0)] = np.nan
    return it.datetime.strftime("%Y-%m"), it.datetime.date().isoformat(), v


cat = Client.open("https://earth-search.aws.element84.com/v1")
items = list(cat.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime="2025-10-01/2026-09-30",
                        query={"grid:code": {"eq": TILE}, "eo:cloud_cover": {"lt": 80}}).items())
items = list({it.datetime.date(): it for it in items}.values())  # una escena por fecha
print(f"{len(items)} escenas del {TILE}, oct 2025 a sep 2026", flush=True)

meses = {}
with ThreadPoolExecutor(8) as ex:
    for mes, dia, v in ex.map(ndvi, items):
        meses[mes] = v if mes not in meses else np.fmax(meses[mes], v)
        print(f"  {dia}: {100 * np.isfinite(v).mean():.0f} % válido", flush=True)

cubo = np.stack([meses[m] for m in sorted(meses)])
n = np.isfinite(cubo).sum(0)
with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
    sup.filter(RuntimeWarning)
    mx, mn = np.nanmax(cubo, 0), np.nanmin(cubo, 0)
din = np.where(n < 3, 0, np.where(mx <= 0.3, 1, np.where(mn > 0.3, 3, 2))).astype("uint8")
np.savez_compressed(SAL / "dinamica_curahuasi.npz", din=din, meses_validos=n.astype("uint8"),
                    ndvi_max=mx.astype("float32"), ndvi_min=mn.astype("float32"), meses=np.array(sorted(meses)))

# --- cruce con lo quemado (dNBR >= 0,10 sembrado) y con la distancia a la cicatriz
quem, san = z["sev"] > 0, z["san"]
d = distance_transform_edt(~quem) * abs(TR[0])
ha = TR[0] * TR[0] / 1e4
grupos = {"agricola_en_el_recuadro": san, "agricola_quemada": san & quem,
          "agricola_sin_quemar_a_1km": san & ~quem & (d <= 1000), "agricola_sin_quemar_1_a_5km": san & ~quem & (d > 1000) & (d <= 5000)}
res = {"meses": sorted(meses), "escenas": len(items),
       **{g: {c: round(float((m & (din == k)).sum() * ha)) for k, c in enumerate(CLASES)} for g, m in grupos.items()}}
(SAL / "dinamica_resumen.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=2, ensure_ascii=False))
