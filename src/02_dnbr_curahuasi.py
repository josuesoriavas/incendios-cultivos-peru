"""
Paso 2: área quemada de Curahuasi (Apurímac) con Sentinel-2 L2A, y qué se quemó (WorldCover 2021 y SAN 2024).

Descarga sin GEE (reutiliza el método de Clasificacion_palta_sin_GEE): catálogo STAC de Earth Search, solo la ventana
de la zona, y por cada fecha candidata se mide el % de píxeles válidos EN LA ZONA según la SCL (no la nubosidad de la escena).
Índices a 20 m: NBR = (B8A-B12)/(B8A+B12); dNBR = NBR_antes - NBR_después; RBR = dNBR / (NBR_antes + 1.001) (Parks et al. 2014).
Máscara: SCL (sin dato, saturado, sombra, nube, cirro, agua) y MNDWI = (B3-B11)/(B3+B11) > 0 en cualquiera de las dos fechas.
Zona: recuadro del incendio E41216 del monitor del profesor. dNBR sembrado con focos: cuenta cada mancha continua
de dNBR >= 0,10 que toque algún píxel a <= 1 km de un foco VIIRS.
Severidad (Key y Benson 2006): baja 0.10-0.27, moderada baja 0.27-0.44, moderada alta 0.44-0.66, alta >= 0.66.

Uso:  uv run python src/02_dnbr_curahuasi.py
"""
import json
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from scipy.ndimage import binary_dilation, label

RAIZ = Path(__file__).resolve().parent.parent
SAL = RAIZ / "data/processed/curahuasi"; SAL.mkdir(parents=True, exist_ok=True)
MAPAS = RAIZ / "outputs/maps"
SAN = r"C:/Users/Josue/Documents/01_MIDAGRI/02_Observatorio_riesgo_inundacion/Proyecto El Niño/Superficie Agrícola/SAN2024.gpkg"
WC = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_S15W075_Map.tif"
BBOX = (-72.57671, -13.72427, -72.3556, -13.50937)  # E41216 del monitor del profesor
VENTANAS = {"antes": "2026-09-20/2026-10-06", "despues": "2026-10-12/2026-10-21"}
SCL_MALOS = [0, 1, 3, 6, 8, 9, 10]  # 6 = agua
CORTES = [0.10, 0.27, 0.44, 0.66]
CLASES = ["Baja", "Moderada baja", "Moderada alta", "Alta"]
WC_NOMBRES = {10: "Bosque", 20: "Matorral", 30: "Pastizal", 40: "Cultivo", 50: "Urbano", 60: "Suelo desnudo",
              70: "Nieve", 80: "Agua", 90: "Humedal", 95: "Manglar", 100: "Musgo"}
env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")


def leer(href, forma=None, bbox=BBOX):
    """Lee solo la ventana del bbox de un COG remoto, remuestreada a `forma` si se pide."""
    for intento in range(4):
        try:
            with env, rasterio.open(href) as src:
                b = transform_bounds(4326, src.crs, *bbox)
                w = from_bounds(*b, src.transform).round_offsets().round_lengths()
                a = src.read(1, window=w, out_shape=forma, resampling=Resampling.bilinear if forma else Resampling.nearest)
                tr = src.window_transform(w)
                if forma:
                    tr = tr * tr.scale(w.width / forma[1], w.height / forma[0])
                return a, tr, src.crs
        except Exception as ex:  # errores transitorios de red
            print(f"    reintento {intento + 1}: {ex}", flush=True); time.sleep(3)
    raise RuntimeError(href)


# --- 1. elegir la mejor escena antes y después: mismo tile, mayor % válido en la zona
cat = Client.open("https://earth-search.aws.element84.com/v1")
cand = {k: list(cat.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime=v, query={"eo:cloud_cover": {"lt": 80}}).items())
        for k, v in VENTANAS.items()}
val = {}
for k, items in cand.items():
    for it in items:
        scl, _, _ = leer(it.assets["scl"].href)
        val[it.id] = (k, it.properties["grid:code"], it.datetime.date().isoformat(), 100 * (~np.isin(scl, SCL_MALOS)).mean(), it)
        print(f"  {k:8s} {it.datetime.date()} {it.properties['grid:code']} nubes escena {it.properties['eo:cloud_cover']:.0f}% | válido en zona {val[it.id][3]:.0f}%", flush=True)
tiles = {v[1] for v in val.values()}
mejor = max(((t, max((v for v in val.values() if v[0] == "antes" and v[1] == t), key=lambda v: v[3], default=None),
                 max((v for v in val.values() if v[0] == "despues" and v[1] == t), key=lambda v: v[3], default=None)) for t in tiles),
            key=lambda x: min(x[1][3] if x[1] else 0, x[2][3] if x[2] else 0))
tile, A, D = mejor
print(f"elegidas: tile {tile} | antes {A[2]} ({A[3]:.0f}% válido) | después {D[2]} ({D[3]:.0f}% válido)", flush=True)


def bandas(it):
    scl, tr, crs = leer(it.assets["scl"].href)
    f = scl.shape
    r = {b: leer(it.assets[b].href, forma=f)[0].astype("float32") * 1e-4 for b in ["red", "green", "blue", "nir08", "swir16", "swir22"]}
    nbr = (r["nir08"] - r["swir22"]) / (r["nir08"] + r["swir22"])
    mndwi = (r["green"] - r["swir16"]) / (r["green"] + r["swir16"])
    malo = np.isin(scl, SCL_MALOS) | (r["nir08"] == 0)
    rgb = np.stack([r["red"], r["green"], r["blue"]])  # color real, para el visor web
    return nbr, mndwi, malo, tr, crs, rgb


nbr_a, mw_a, malo_a, tr, crs, rgb_a = bandas(A[4])
nbr_d, mw_d, malo_d, _, _, rgb_d = bandas(D[4])
agua = (mw_a > 0) | (mw_d > 0)
sin_dato = malo_a | malo_d
dnbr = nbr_a - nbr_d
rbr = dnbr / (nbr_a + 1.001)
H, W = dnbr.shape; px_ha = abs(tr.a * tr.e) / 1e4

# --- 2. zona del incendio: <= 1 km de los focos (los del monitor desde el 6 oct + los FIRMS frescos)
ev = json.loads((RAIZ / "data/raw/monitor_profesor/eventos/E41216.json").read_text(encoding="utf-8"))
f_prof = gpd.GeoSeries(gpd.points_from_xy([p[0] for p in ev["focos"]], [p[1] for p in ev["focos"]]), crs=4326)
f_new = gpd.read_file(RAIZ / "data/processed/focos.gpkg", bbox=BBOX).geometry if (RAIZ / "data/processed/focos.gpkg").exists() else gpd.GeoSeries(crs=4326)
focos = gpd.GeoSeries(list(f_prof) + list(f_new.to_crs(4326)), crs=4326).to_crs(crs)
zona = rasterize(((p.buffer(1000), 1) for p in focos), out_shape=(H, W), transform=tr, fill=0, dtype="uint8").astype(bool)

# dNBR sembrado con focos: se conserva cada mancha continua de dNBR >= 0,10 que toque la zona de algún foco
valido = ~sin_dato & ~agua
etq, _ = label(valido & (dnbr >= CORTES[0]))
sembradas = np.setdiff1d(np.unique(etq[zona]), [0])
ok = valido & np.isin(etq, sembradas)
sev = np.zeros((H, W), "uint8")
for i, c in enumerate(CORTES):
    sev[ok & (dnbr >= c)] = i + 1
quem = sev > 0

# --- 3. qué se quemó: WorldCover 2021 (10 m -> moda a 20 m) y SAN 2024
wc10, wtr, wcrs = leer(WC)
wc = np.zeros((H, W), "uint8")
reproject(wc10, wc, src_transform=wtr, src_crs=wcrs, dst_transform=tr, dst_crs=crs, resampling=Resampling.mode)
san = gpd.read_file(SAN, bbox=BBOX, columns=["IDDIST"]).to_crs(crs)
san_m = rasterize(((g_, 1) for g_ in san.geometry), out_shape=(H, W), transform=tr, fill=0, dtype="uint8").astype(bool)

ha = lambda m: round(float(m.sum() * px_ha))
res = {
    "tile": tile, "antes": A[2], "despues": D[2], "valido_zona_antes_pct": round(A[3]), "valido_zona_despues_pct": round(D[3]),
    "pct_sin_dato_en_zona": round(100 * (zona & sin_dato).sum() / zona.sum(), 1), "agua_excluida_ha": ha(zona & agua & ~sin_dato),
    "quemado_dnbr_010_ha": ha(quem), "quemado_dnbr_027_ha": ha(ok & (dnbr >= 0.27)),
    "quemado_rbr_010_ha": ha(ok & (rbr >= 0.10)), "solo_a_1km_de_focos_ha": ha(quem & zona),
    "severidad_ha": {CLASES[i]: ha(sev == i + 1) for i in range(4)},
    "worldcover_en_quemado_ha": {WC_NOMBRES.get(int(k), str(k)): ha(quem & (wc == k)) for k in np.unique(wc[quem])},
    "san2024_en_quemado_ha": ha(quem & san_m),
    "san2024_en_quemado_por_worldcover_ha": {WC_NOMBRES.get(int(k), str(k)): ha(quem & san_m & (wc == k)) for k in np.unique(wc[quem & san_m])},
    "san2024_a_5km_sin_quemar_ha": ha(binary_dilation(quem, iterations=250) & san_m & ~quem),
    "monitor_profesor_ha": ev["severidad"]["total_ha"], "monitor_profesor_fechas": ev["sentinel"],
}
np.savez_compressed(SAL / "capas_curahuasi.npz", dnbr=dnbr, rbr=rbr, sev=sev, sembrado=ok, zona=zona, sin_dato=sin_dato, agua=agua,
                    wc=wc, san=san_m, rgb_antes=rgb_a, rgb_despues=rgb_d, transform=np.array(tuple(tr)[:6]), crs=str(crs))
(SAL / "resumen_curahuasi.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
with rasterio.open(SAL / "dnbr_curahuasi.tif", "w", driver="GTiff", height=H, width=W, count=2, dtype="float32", crs=crs, transform=tr, compress="deflate") as d:
    d.write(np.where(sin_dato, np.nan, dnbr), 1); d.write(np.where(sin_dato, np.nan, rbr), 2)
print(json.dumps(res, indent=2, ensure_ascii=False))

# --- 4. mapa
ext = [tr.c, tr.c + W * tr.a, tr.f + H * tr.e, tr.f]
fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
ax[0].imshow(np.where(sin_dato, np.nan, dnbr), cmap="RdYlGn_r", vmin=-0.3, vmax=0.8, extent=ext)
ax[0].set_title(f"dNBR  ({A[2]} → {D[2]}, Sentinel-2 {tile})")
fondo = np.where(sin_dato, 0, 1)
ax[1].imshow(fondo, cmap=ListedColormap(["#cccccc", "#f4f1ea"]), extent=ext)
ax[1].imshow(np.ma.masked_where(~san_m, san_m), cmap=ListedColormap(["#5b8c3a"]), alpha=0.55, extent=ext)
ax[1].imshow(np.ma.masked_where(sev == 0, sev), cmap=ListedColormap(["#fdd49e", "#fc8d59", "#d7301f", "#7f0000"]), vmin=1, vmax=4, extent=ext)
ax[1].set_title(f"Quemado {res['quemado_dnbr_010_ha']:,} ha (dNBR ≥ 0,10) · cultivo SAN 2024 quemado {res['san2024_en_quemado_ha']:,} ha")
from matplotlib.patches import Patch
ax[1].legend(handles=[Patch(color=c, label=l) for c, l in zip(["#fdd49e", "#fc8d59", "#d7301f", "#7f0000", "#5b8c3a", "#cccccc"], CLASES + ["Superficie agrícola 2024 (MIDAGRI)", "Nube / sin dato"])],
             loc="lower left", fontsize=8)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle("Curahuasi (Abancay, Apurímac): área quemada y superficie agrícola", fontsize=14)
fig.tight_layout(); fig.savefig(MAPAS / "curahuasi_dnbr_cultivo.png", dpi=130)
print("mapa:", MAPAS / "curahuasi_dnbr_cultivo.png")
