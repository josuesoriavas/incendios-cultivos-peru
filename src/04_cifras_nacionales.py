"""
Paso 4: cifras nacionales corregidas.

1. WorldCover 2021 (10 m) en el centro de los MISMOS focos que se cruzaron con la SAN 2024, para comparar ambas fuentes
   sobre los mismos puntos.
2. Superficie agrícola cerca de los incendios activos como UNIÓN de las zonas (huella, anillos de 1 y 5 km): una hectárea
   cerca de varios incendios cuenta una sola vez.

Salidas: data/processed/focos_worldcover.csv y resumen_nacional.json; actualiza data/monitor/resumen.csv, incendios.csv
y departamentos.csv.
Uso:  uv run python src/04_cifras_nacionales.py
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely

RAIZ = Path(__file__).resolve().parent.parent
P, M = RAIZ / "data/processed", RAIZ / "data/monitor"
SAN = r"C:/Users/Josue/Documents/01_MIDAGRI/02_Observatorio_riesgo_inundacion/Proyecto El Niño/Superficie Agrícola/SAN2024.gpkg"
UTM = 32718
WC_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif"
WC = {10: "Bosque", 20: "Matorral", 30: "Pastizal", 40: "Cultivo", 50: "Urbano", 60: "Suelo desnudo", 70: "Nieve",
      80: "Agua", 90: "Humedal", 95: "Manglar", 100: "Musgo"}

foc = gpd.read_file(P / "focos.gpkg")
inc = gpd.read_file(P / "incendios.gpkg")
foc["en_cultivo_san"] = foc.en_cultivo_san.astype(bool)
inc["activo"] = inc.activo.astype(bool)


# --- 1. WorldCover en el centro de cada foco (tiles de 3 x 3 grados, nombrados por su esquina suroeste)
def tile(lat, lon):
    la, lo = int(np.floor(lat / 3) * 3), int(np.floor(lon / 3) * 3)
    return f"{'S' if la < 0 else 'N'}{abs(la):02d}{'W' if lo < 0 else 'E'}{abs(lo):03d}"


foc["tile"] = [tile(a, b) for a, b in zip(foc.latitude, foc.longitude)]


def muestrear(t):
    sub = foc[foc.tile == t].sort_values(["latitude", "longitude"])  # ordenados: reaprovecha los bloques leídos
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2"), \
            rasterio.open(WC_URL.format(t)) as src:
        return pd.Series([int(v[0]) for v in src.sample(zip(sub.longitude, sub.latitude))], index=sub.index)


previo = pd.read_csv(P / "focos_worldcover.csv") if (P / "focos_worldcover.csv").exists() else None
if previo is not None and len(previo) == len(foc) and np.allclose(previo.latitude, foc.latitude) and np.allclose(previo.longitude, foc.longitude):
    foc["wc"] = previo.wc.to_numpy()  # mismos focos en el mismo orden: reutiliza la lectura anterior
else:
    with ThreadPoolExecutor(8) as ex:
        foc["wc"] = pd.concat(list(ex.map(muestrear, sorted(foc.tile.unique()))))
foc["wc_clase"] = foc.wc.map(WC).fillna("Sin dato")
foc[["incendio", "latitude", "longitude", "en_cultivo_san", "wc", "wc_clase"]].to_csv(P / "focos_worldcover.csv", index=False)
pct_san = 100 * foc.en_cultivo_san.mean()
pct_wc = 100 * (foc.wc == 40).mean()
wc_en_san = (100 * foc[foc.en_cultivo_san].wc_clase.value_counts(normalize=True)).round(1)
print(f"focos: {len(foc)} | en SAN 2024: {pct_san:.1f} % | en Cultivo de WorldCover: {pct_wc:.1f} %", flush=True)
print("WorldCover de los focos en SAN 2024 (%):", wc_en_san.to_dict(), flush=True)
dominante = foc.groupby("incendio").wc_clase.agg(lambda s: s.value_counts().index[0])

# --- 2. superficie agrícola en la UNIÓN de las zonas alrededor de los incendios activos
dist = gpd.read_file(RAIZ / "data/raw/limites/peru_distrital_simple.geojson")
dep_nombre = dist.drop_duplicates("IDDPTO").set_index("IDDPTO").NOMBDEP
act = inc[inc.activo].to_crs(UTM)
zonas = {"huella": act.geometry.union_all(), "1km": act.buffer(1000).union_all(), "5km": act.buffer(5000).union_all()}
mask = gpd.GeoSeries([zonas["5km"]], crs=UTM).to_crs(4326).iloc[0]
san = gpd.read_file(SAN, mask=mask, columns=["IDDPTO", "IDDIST"])
print(f"SAN 2024: {len(san)} distritos cerca de incendios activos", flush=True)
san = san.explode(ignore_index=True).to_crs(UTM)
san["geometry"] = san.geometry.make_valid()


def ha_por_departamento(zona):
    """Hectáreas de SAN 2024 dentro de la zona, por departamento. Cada parte de la unión se procesa sola."""
    tot = pd.Series(dtype=float)
    for parte in gpd.GeoSeries([zona], crs=UTM).explode(index_parts=False):
        shapely.prepare(parte)
        idx = san.sindex.query(parte, predicate="intersects")
        if not len(idx):
            continue
        g = san.geometry.iloc[idx].to_numpy()
        dentro = shapely.contains_properly(parte, g)
        a = shapely.area(g)
        if (~dentro).any():
            a[~dentro] = shapely.area(shapely.intersection(g[~dentro], parte))
        tot = tot.add(pd.Series(a / 1e4).groupby(san.IDDPTO.to_numpy()[idx]).sum(), fill_value=0)
    return tot.rename(index=dep_nombre).sort_values(ascending=False)


ha = {k: ha_por_departamento(v) for k, v in zonas.items()}
h5 = ha["5km"]
print(f"agrícola a <= 5 km de activos: unión {h5.sum():,.0f} ha (antes, sumando incendios: {act.cultivo_5km_ha.sum():,.0f} ha)", flush=True)

# --- salidas
res = {
    "focos": len(foc), "pct_focos_agricola_midagri": round(pct_san, 1), "pct_focos_cultivo_worldcover": round(pct_wc, 1),
    "worldcover_de_focos_en_san_pct": wc_en_san.to_dict(),
    "activos": int(len(act)),
    "agricola_huella_activos_ha": round(ha["huella"].sum()), "agricola_1km_activos_ha": round(ha["1km"].sum()),
    "agricola_5km_activos_ha": round(h5.sum()), "agricola_5km_suma_por_incendio_ha": round(act.cultivo_5km_ha.sum()),
    "agricola_5km_por_departamento_ha": h5.round().head(8).to_dict(),
}
(P / "resumen_nacional.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

r = pd.read_csv(M / "resumen.csv").iloc[0].to_dict()
con_agri = int((act.cultivo_en_huella_ha > 0).sum())
top = act.sort_values("cultivo_5km_ha", ascending=False).iloc[0]
r.update(focos=len(foc), activos=int(len(act)), pct_focos_agricola_midagri=round(pct_san, 1),
         activos_con_agricola_en_huella=con_agri, pct_activos_con_agricola=round(100 * con_agri / len(act)),
         top_inc_distrito=top.distrito.title(), top_inc_dep=top.departamento.title(), top_inc_5km_ha=round(top.cultivo_5km_ha))
r.update(pct_focos_cultivo_worldcover=round(pct_wc, 1), wc_vs_midagri_veces=round(pct_san / pct_wc, 1) if pct_wc else "",
         agricola_en_huella_activos_ha=round(ha["huella"].sum()), agricola_1km_activos_ha=round(ha["1km"].sum()),
         agricola_5km_activos_ha=round(h5.sum()), pct_top3_deps=round(100 * h5.iloc[:3].sum() / h5.sum()),
         pct_focos_agricola_wc_pastizal_bosque=round(wc_en_san.get("Pastizal", 0) + wc_en_san.get("Bosque", 0)))
for i in range(3):
    r[f"dep{i + 1}"], r[f"dep{i + 1}_ha"] = h5.index[i].title(), round(h5.iloc[i])
pd.DataFrame([r]).to_csv(M / "resumen.csv", index=False)

a = inc[inc.activo | (inc.huella_ha > 100)].copy()
a["estado"] = np.where(a.activo, "Activo", "Sin focos 24 h")
a["wc_cobertura"] = a.incendio.map(dominante).fillna("Sin dato")
cols = ["incendio", "distrito", "departamento", "estado", "lat", "lon", "focos", "focos_24h", "huella_ha", "cultivo_en_huella_ha",
        "cultivo_1km_ha", "cultivo_5km_ha", "pct_focos_cultivo", "wc_cobertura"]
a[cols].round(2).to_csv(M / "incendios.csv", index=False)

d = pd.read_csv(M / "departamentos.csv")
g = foc.groupby("NOMBDEP")
d["focos"] = d.departamento.map(g.size()).fillna(0).astype(int)
d["focos_agricola"] = d.departamento.map(g.en_cultivo_san.sum()).fillna(0).astype(int)
d["pct_agricola"] = (100 * d.focos_agricola / d.focos).round(1)
d["activos"] = d.departamento.map(act.groupby("departamento").size()).fillna(0).astype(int)
d["agricola_5km_ha"] = d.departamento.map(h5).fillna(0).round()
d["focos_wc_cultivo"] = d.departamento.map(foc[foc.wc == 40].groupby("NOMBDEP").size()).fillna(0).astype(int)
d.to_csv(M / "departamentos.csv", index=False)
print(json.dumps(res, indent=2, ensure_ascii=False))
