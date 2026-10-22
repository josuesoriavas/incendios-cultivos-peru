"""
Paso 1: focos VIIRS (FIRMS, 7 días) -> incendios -> cruce con la Superficie Agrícola Nacional 2024 (MIDAGRI).

Definición de incendio (la misma del monitor del profesor): focos a <= 1 km y <= 48 h entre sí;
"activo" si tiene al menos un foco en las últimas 24 h del corte.
Huella: unión de los píxeles VIIRS (375 m) de sus focos. No es área quemada.

Salidas: data/processed/focos.gpkg, incendios.gpkg, incendios.csv, resumen_paso1.json
Uso:  uv run python src/01_incendios_cultivo.py
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import box
from sklearn.neighbors import BallTree

RAIZ = Path(__file__).resolve().parent.parent
SAL = RAIZ / "data/processed"; SAL.mkdir(parents=True, exist_ok=True)
SAN = r"C:/Users/Josue/Documents/01_MIDAGRI/02_Observatorio_riesgo_inundacion/Proyecto El Niño/Superficie Agrícola/SAN2024.gpkg"
UTM = 32718  # UTM 18S: sirve para medir distancias y áreas en casi todo el Perú
VOLCANES = [(-71.857, -15.787), (-70.903, -16.355), (-71.409, -16.294)]  # Sabancaya, Ubinas, Misti

# --- 1. focos de los tres satélites, solo Perú
f = pd.concat([pd.read_csv(p) for p in sorted((RAIZ / "data/raw/firms").glob("*.csv"))], ignore_index=True)
f = f[f.confidence != "l"]  # fuera los de confianza baja
f["t"] = pd.to_datetime(f.acq_date + " " + f.acq_time.astype(str).str.zfill(4), format="%Y-%m-%d %H%M", utc=True)
dist = gpd.read_file(RAIZ / "data/raw/limites/peru_distrital_simple.geojson")[["IDDIST", "NOMBDEP", "NOMBPROV", "NOMBDIST", "geometry"]]
g = gpd.GeoDataFrame(f, geometry=gpd.points_from_xy(f.longitude, f.latitude), crs=4326)
g = gpd.sjoin(g, dist, predicate="within").drop(columns="index_right")
v = gpd.GeoSeries(gpd.points_from_xy(*zip(*VOLCANES)), crs=4326).to_crs(UTM).buffer(3000).union_all()
g = g[~g.to_crs(UTM).within(v)].reset_index(drop=True)
corte = g.t.max()
print(f"focos en Perú: {len(g)} | {g.t.min():%Y-%m-%d} a {corte:%Y-%m-%d %H:%M} UTC", flush=True)

# --- 2. agrupar: vecinos a <= 1 km y <= 48 h -> componentes conexos
xy = np.radians(g[["latitude", "longitude"]].to_numpy())
vec = BallTree(xy, metric="haversine").query_radius(xy, r=1.0 / 6371.0)
i = np.repeat(np.arange(len(g)), [len(n) for n in vec]); j = np.concatenate(vec)
horas = g.t.astype("int64").to_numpy() / 3.6e12
ok = np.abs(horas[i] - horas[j]) <= 48
_, g["incendio"] = connected_components(coo_matrix((np.ones(ok.sum()), (i[ok], j[ok])), shape=(len(g), len(g))), directed=False)

# --- 3. huella = unión de píxeles de 375 m
gu = g.to_crs(UTM)
gu["geometry"] = [box(p.x - 187.5, p.y - 187.5, p.x + 187.5, p.y + 187.5) for p in gu.geometry]
huella = gu.dissolve("incendio").geometry
ult24 = g.t >= corte - pd.Timedelta(hours=24)
inc = g.groupby("incendio").agg(focos=("t", "size"), inicio=("t", "min"), ultimo=("t", "max"),
                                frp_total=("frp", "sum"), lat=("latitude", "mean"), lon=("longitude", "mean"),
                                departamento=("NOMBDEP", lambda s: s.mode()[0]), distrito=("NOMBDIST", lambda s: s.mode()[0]),
                                iddist=("IDDIST", lambda s: s.mode()[0]))
inc["focos_24h"] = ult24.groupby(g.incendio).sum()
inc["activo"] = inc.focos_24h > 0
inc = gpd.GeoDataFrame(inc, geometry=huella.reindex(inc.index), crs=UTM)
inc["huella_ha"] = inc.area / 1e4
print(f"incendios: {len(inc)} | activos: {inc.activo.sum()} | activos > 100 ha: {(inc.activo & (inc.huella_ha > 100)).sum()}", flush=True)

# --- 4. cruce con SAN 2024 (solo los distritos con focos y sus vecinos, para no leer los 714 MB)
ids = sorted(gpd.sjoin(dist, inc.to_crs(4326).buffer(0.06).to_frame("geometry"), predicate="intersects").IDDIST.unique())
san = gpd.read_file(SAN, where=f"IDDIST IN ({','.join(repr(x) for x in ids)})", columns=["IDDIST", "AREA_HA"])
print(f"SAN 2024 leída: {len(san)} distritos, {san.AREA_HA.sum():,.0f} ha agrícolas", flush=True)
san = san.explode(ignore_index=True).to_crs(UTM)  # parcelas sueltas: el índice espacial trabaja con piezas chicas
san["geometry"] = san.geometry.make_valid()
print(f"  {len(san)} parcelas", flush=True)

pts = gpd.GeoDataFrame(geometry=gu.centroid, crs=UTM)
g["en_cultivo_san"] = g.index.isin(gpd.sjoin(pts, san[["geometry"]], predicate="within").index)
inc["focos_en_cultivo"] = g.groupby("incendio").en_cultivo_san.sum()
inc["pct_focos_cultivo"] = 100 * inc.focos_en_cultivo / inc.focos


def ha_cultivo(geoms):
    """Hectáreas de SAN 2024 dentro de cada geometría (solo las parcelas candidatas del índice espacial)."""
    out = []
    for geom in geoms:
        cand = san.geometry.iloc[san.sindex.query(geom, predicate="intersects")]
        out.append(cand.intersection(geom).area.sum() / 1e4 if len(cand) else 0.0)
    return out


rel = inc.activo | (inc.huella_ha > 100)  # anillos solo para los incendios relevantes
inc["cultivo_en_huella_ha"] = ha_cultivo(inc.geometry)
for km in (1, 5):
    inc[f"cultivo_{km}km_ha"] = np.nan
    inc.loc[rel, f"cultivo_{km}km_ha"] = ha_cultivo(inc.geometry[rel].buffer(km * 1000))
print("cruce con SAN listo", flush=True)

# --- 5. comparación con WorldCover según el monitor del profesor (corte 17 oct): su incendio más cercano a <= 3 km
e = json.loads((RAIZ / "data/raw/monitor_profesor/eventos.json").read_text(encoding="utf-8"))
p = pd.DataFrame(e["filas"], columns=e["cols"])
pg = gpd.GeoDataFrame(p[["evento", "cobertura", "huella_ha", "prioridad"]].add_prefix("prof_"),
                      geometry=gpd.points_from_xy(p.lon, p.lat), crs=4326).to_crs(UTM)
c = gpd.GeoDataFrame(geometry=inc.centroid, crs=UTM)
m = gpd.sjoin_nearest(c, pg, max_distance=3000, how="left").groupby(level=0).first()
inc = inc.join(m.drop(columns=["geometry", "index_right"]))

# --- salidas
g.to_crs(4326).to_file(SAL / "focos.gpkg", driver="GPKG")
inc.to_crs(4326).to_file(SAL / "incendios.gpkg", driver="GPKG")
cols = ["departamento", "distrito", "iddist", "activo", "focos", "focos_24h", "huella_ha", "cultivo_en_huella_ha",
        "pct_focos_cultivo", "cultivo_1km_ha", "cultivo_5km_ha", "prof_cobertura", "prof_evento", "inicio", "ultimo", "lat", "lon"]
inc.sort_values("huella_ha", ascending=False)[cols].round(1).to_csv(SAL / "incendios.csv", encoding="utf-8-sig")

act = inc[inc.activo]
res = {
    "corte_utc": f"{corte:%Y-%m-%d %H:%M}", "desde_utc": f"{g.t.min():%Y-%m-%d}",
    "focos": len(g), "incendios": len(inc), "activos": int(len(act)),
    "pct_focos_en_cultivo_san": round(100 * g.en_cultivo_san.mean(), 1),
    "pct_focos_cultivo_worldcover_monitor": 1,
    "activos_con_cultivo_en_huella": int((act.cultivo_en_huella_ha > 0).sum()),
    "cultivo_en_huella_activos_ha": round(act.cultivo_en_huella_ha.sum()),
    "cultivo_a_5km_activos_ha": round(act.cultivo_5km_ha.sum()),
    "prof_cobertura_de_incendios_con_cultivo_san": inc[inc.cultivo_en_huella_ha > 0].prof_cobertura.value_counts().to_dict(),
    "por_departamento_cultivo_5km_activos": act.groupby("departamento").cultivo_5km_ha.sum().sort_values(ascending=False).round().head(8).to_dict(),
}
(SAL / "resumen_paso1.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=2, ensure_ascii=False))
print(inc.sort_values("huella_ha", ascending=False)[cols[:12]].head(12).round(1).to_string())
