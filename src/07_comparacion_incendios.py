"""
Paso 7: el mismo análisis de Curahuasi para los cinco incendios más grandes de octubre de 2026 después de Curahuasi
(se excluye La Joya, que es suelo desnudo y probablemente una fuente fija).

Por incendio, en su recuadro y en la grilla de 20 m de Sentinel-2:
  1. imágenes antes y después del fuego elegidas por el % de píxeles válidos en la zona (capa SCL), un solo tile MGRS
  2. dNBR sembrado con focos VIIRS (umbral 0,10; severidad de Key y Benson) y máscara de nubes, sombras y agua
  3. cruce con la SAN 2024 (MIDAGRI), ESA WorldCover 2021 y la distancia euclidiana a la cicatriz
  4. NDVI máximo mensual de oct 2025 a sep 2026 (hasta 4 escenas por mes)
  5. qué hay dentro de lo agrícola (cultivo, pasto o matorral, árboles): Random Forest como el del paso 6, pero entrenado
     con las referencias de los seis recuadros juntos (incluido Curahuasi) y validado por bloques de 2 km
Curahuasi conserva los resultados de los pasos 2, 5 y 6.
Salidas: data/processed/comparacion/, data/monitor/comparacion.csv y data/monitor/miniaturas.json
Uso:  uv run python src/07_comparacion_incendios.py
"""
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt, label, uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupKFold

RAIZ = Path(__file__).resolve().parent.parent
EV = RAIZ / "data/raw/monitor_profesor/eventos"
SAL = RAIZ / "data/processed/comparacion"; SAL.mkdir(parents=True, exist_ok=True)
MON = RAIZ / "data/monitor"
SAN = r"C:/Users/Josue/Documents/01_MIDAGRI/02_Observatorio_riesgo_inundacion/Proyecto El Niño/Superficie Agrícola/SAN2024.gpkg"
INCENDIOS = {"E45347": "Río Tambo", "E38662": "Sicuani", "E37866": "Coyllurqui", "E41197": "San Pedro de Cachora", "E39747": "Alto Pichigua"}
SLUGS = {"E45347": "riotambo", "E38662": "sicuani", "E37866": "coyllurqui", "E41197": "cachora", "E39747": "altopichigua"}
SCL_MALOS = [0, 1, 3, 6, 8, 9, 10]  # sin dato, saturado, sombra, agua, nubes, cirros
SCL_NUBES = [0, 1, 3, 8, 9, 10]
CORTES = [0.10, 0.27, 0.44, 0.66]
CLASES = ["cultivo", "pasto o matorral", "árboles"]
HOY = date(2026, 10, 21)
cat = Client.open("https://earth-search.aws.element84.com/v1")
GDAL = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")


def leer(href, bbox, forma=None):
    """Ventana del recuadro en un COG remoto (remuestreada a `forma` si se pide)."""
    for intento in range(4):
        try:
            with rasterio.Env(**GDAL), rasterio.open(href) as src:
                w = from_bounds(*transform_bounds(4326, src.crs, *bbox), src.transform).round_offsets().round_lengths()
                a = src.read(1, window=w, out_shape=forma, boundless=True, fill_value=0,
                             resampling=Resampling.bilinear if forma else Resampling.nearest)
                return a, src.window_transform(w), src.crs
        except Exception as ex:
            print(f"    reintento {intento + 1}: {ex}", flush=True); time.sleep(3)
    raise RuntimeError(href)


def a_grilla(urls, bbox, forma, tr, crs, resampling, tipo):
    """Lleva uno o varios tiles en grados (WorldCover, DEM) a la grilla del recuadro."""
    out = np.zeros(forma, tipo)
    for u in urls:
        try:
            with rasterio.Env(**GDAL), rasterio.open(u) as s:
                m = 0.02
                w = from_bounds(bbox[0] - m, bbox[1] - m, bbox[2] + m, bbox[3] + m, s.transform).round_offsets().round_lengths()
                a = s.read(1, window=w, boundless=True, fill_value=0).astype(tipo)
                tmp = np.zeros(forma, tipo)
                reproject(a, tmp, src_transform=s.window_transform(w), src_crs=s.crs, dst_transform=tr, dst_crs=crs,
                          resampling=resampling, src_nodata=0, dst_nodata=0)
                out = np.where(tmp != 0, tmp, out)
        except rasterio.errors.RasterioIOError:
            pass  # tile inexistente (océano)
    return out


def esquinas(bbox, paso):
    return {(int(np.floor(la / paso) * paso), int(np.floor(lo / paso) * paso)) for la in (bbox[1], bbox[3]) for lo in (bbox[0], bbox[2])}


def url_wc(la, lo):
    return f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{'S' if la < 0 else 'N'}{abs(la):02d}{'W' if lo < 0 else 'E'}{abs(lo):03d}_Map.tif"


def url_dem(la, lo):
    n = f"Copernicus_DSM_COG_10_{'S' if la < 0 else 'N'}{abs(la):02d}_00_{'W' if lo < 0 else 'E'}{abs(lo):03d}_00_DEM"
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"


def variables(cubo, el, tr):
    """Las mismas variables del paso 6: 12 meses, mínimo, máximo, amplitud, mes del pico, lluvias, secas, altitud, pendiente."""
    with np.errstate(all="ignore"):
        med = np.nanmedian(cubo, 0)
    c = np.where(np.isfinite(cubo), cubo, med)
    gy, gx = np.gradient(el, abs(tr.a))
    pend = np.degrees(np.arctan(np.hypot(gx, gy)))
    return np.column_stack([c.reshape(12, -1).T, c.min(0).ravel(), c.max(0).ravel(), (c.max(0) - c.min(0)).ravel(), c.argmax(0).ravel(),
                            c[3:6].mean(0).ravel(), c[8:11].mean(0).ravel(), el.ravel(), pend.ravel()]).astype("float32")


def referencias(san, wc, X, tr):
    puro = lambda m: uniform_filter(m.astype("float32"), 3).ravel() > 0.999
    lejos = (distance_transform_edt(~san) * abs(tr.a)).ravel() > 200
    ref = np.full(san.size, -1, "int8")
    ref[puro(san & (wc == 40))] = 0
    ref[~san.ravel() & lejos & puro(np.isin(wc, [20, 30]))] = 1
    ref[~san.ravel() & lejos & puro(wc == 10)] = 2
    ref[~np.isfinite(X).all(1)] = -1
    return ref


def mejor_par(bbox, ini, fin_pre, ini_post):
    cand = {}
    for k, v in {"antes": f"{ini}/{fin_pre}", "despues": f"{ini_post}/{HOY}"}.items():
        for it in cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=v, query={"eo:cloud_cover": {"lt": 80}}).items():
            scl, _, _ = leer(it.assets["scl"].href, bbox)
            cand.setdefault((k, it.properties["grid:code"]), []).append((100 * (~np.isin(scl, SCL_MALOS)).mean(), it.datetime.date(), it))
    mejor = None
    for t in {t for _, t in cand}:
        a, d = max(cand.get(("antes", t), [(0, None, None)]), key=lambda x: (x[0], x[1] or date.min)), \
               max(cand.get(("despues", t), [(0, None, None)]), key=lambda x: (x[0], x[1] or date.min))
        if a[2] and d[2] and (mejor is None or min(a[0], d[0]) > min(mejor[1][0], mejor[2][0])):
            mejor = (t, a, d)
    return mejor


def procesar(eid):
    d = json.loads((EV / f"{eid}.json").read_text(encoding="utf-8"))
    bbox = tuple(d["bounds"])
    ev = EVENTOS.loc[eid]
    inicio, ultimo = date.fromisoformat(ev.inicio[:10]), date.fromisoformat(ev.ultimo[:10])
    tile, A, D = mejor_par(bbox, inicio - timedelta(days=25), inicio - timedelta(days=1), inicio + timedelta(days=3))
    print(f"{INCENDIOS[eid]}: tile {tile} | antes {A[1]} ({A[0]:.0f} %) | después {D[1]} ({D[0]:.0f} %)", flush=True)

    def bandas(it):
        scl, tr, crs = leer(it.assets["scl"].href, bbox)
        f = scl.shape
        r = {b: leer(it.assets[b].href, bbox, f)[0].astype("float32") for b in ["green", "nir08", "swir16", "swir22"]}
        with np.errstate(all="ignore"):
            nbr = (r["nir08"] - r["swir22"]) / (r["nir08"] + r["swir22"])
            mndwi = (r["green"] - r["swir16"]) / (r["green"] + r["swir16"])
        return nbr, mndwi, np.isin(scl, SCL_MALOS) | (r["nir08"] == 0), tr, crs

    nbr_a, mw_a, malo_a, tr, crs = bandas(A[2])
    nbr_d, mw_d, malo_d, _, _ = bandas(D[2])
    H, W = nbr_a.shape

    def color(it):  # color real a 40 m para el visor web
        rgb = np.stack([leer(it.assets[b].href, bbox, (H // 2, W // 2))[0].astype("float32") * 1e-4 for b in ("red", "green", "blue")])
        return (np.clip(rgb / 0.25, 0, 1) ** 0.8 * 255).astype("uint8")

    rgb_a, rgb_d = color(A[2]), color(D[2])
    sin_dato, agua = malo_a | malo_d, (mw_a > 0) | (mw_d > 0)
    dnbr = np.where(sin_dato, np.nan, nbr_a - nbr_d)

    # focos: los del incendio desde su inicio + los de FIRMS del 14 al 21 oct en el recuadro
    f1 = gpd.GeoSeries(gpd.points_from_xy([p[0] for p in d["focos"]], [p[1] for p in d["focos"]]), crs=4326)
    f2 = gpd.read_file(RAIZ / "data/processed/focos.gpkg", bbox=bbox).geometry
    focos = gpd.GeoSeries(list(f1) + list(f2.to_crs(4326)), crs=4326).to_crs(crs)
    zona = rasterize(((p.buffer(1000), 1) for p in focos), out_shape=(H, W), transform=tr, fill=0, dtype="uint8").astype(bool)
    valido = ~sin_dato & ~agua
    etq, _ = label(valido & (np.nan_to_num(dnbr, nan=-9) >= CORTES[0]))
    quem = np.isin(etq, np.setdiff1d(np.unique(etq[zona]), [0]))
    q027 = quem & (dnbr >= 0.27)

    wc = a_grilla([url_wc(*e) for e in esquinas(bbox, 3)], bbox, (H, W), tr, crs, Resampling.mode, "uint8")
    el = a_grilla([url_dem(*e) for e in esquinas(bbox, 1)], bbox, (H, W), tr, crs, Resampling.bilinear, "float32")
    s = gpd.read_file(SAN, bbox=bbox, columns=["IDDIST"]).to_crs(crs)
    san = rasterize(((g, 1) for g in s.geometry), out_shape=(H, W), transform=tr, fill=0, dtype="uint8").astype(bool) if len(s) else np.zeros((H, W), bool)

    # NDVI máximo mensual de los 12 meses previos (hasta 4 escenas por mes, las de menos nubes)
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime="2025-10-01/2026-09-30",
                            query={"grid:code": {"eq": tile}, "eo:cloud_cover": {"lt": 80}}).items())
    por_mes = {}
    for it in {it.datetime.date(): it for it in items}.values():
        por_mes.setdefault(it.datetime.strftime("%Y-%m"), []).append(it)
    elegidas = [it for m in por_mes.values() for it in sorted(m, key=lambda x: x.properties["eo:cloud_cover"])[:4]]

    def ndvi(it):
        scl, _, _ = leer(it.assets["scl"].href, bbox)
        red = leer(it.assets["red"].href, bbox, (H, W))[0].astype("float32")
        nir = leer(it.assets["nir"].href, bbox, (H, W))[0].astype("float32")
        with np.errstate(all="ignore"):
            v = (nir - red) / (nir + red)
        v[np.isin(scl, SCL_NUBES) | (nir == 0) | (red == 0) | (scl.shape != (H, W))] = np.nan
        return it.datetime.strftime("%Y-%m"), v

    meses = {}
    with ThreadPoolExecutor(8) as ex:
        for m, v in ex.map(ndvi, elegidas):
            meses[m] = v if m not in meses else np.fmax(meses[m], v)
    todos = [f"{y}-{mm:02d}" for y, mm in [(2025, 10), (2025, 11), (2025, 12)] + [(2026, k) for k in range(1, 10)]]
    cubo = np.stack([meses.get(m, np.full((H, W), np.nan, "float32")) for m in todos])
    X = variables(cubo, el, tr)
    print(f"   {len(elegidas)} escenas de NDVI; meses con dato: {sum(m in meses for m in todos)}", flush=True)
    np.savez_compressed(SAL / f"{eid}.npz", quem=quem, q027=q027, dnbr=np.nan_to_num(dnbr, nan=-9).astype("float16"), sin_dato=sin_dato,
                        zona=zona, san=san, wc=wc, el=el, transform=np.array(tuple(tr)[:6]), crs=str(crs))
    return {"eid": eid, "nombre": INCENDIOS[eid], "departamento": ev.departamento, "tr": tr, "H": H, "W": W, "quem": quem, "q027": q027,
            "sin_dato": sin_dato, "zona": zona, "san": san, "wc": wc, "el": el, "X": X, "ref": referencias(san, wc, X, tr),
            "agua": agua, "dnbr": dnbr, "rgb_antes": rgb_a, "rgb_despues": rgb_d, "crs": str(crs),
            "antes": A[1].isoformat(), "despues": D[1].isoformat(), "valido_antes": round(A[0]), "valido_despues": round(D[0])}


e = json.loads((RAIZ / "data/raw/monitor_profesor/eventos.json").read_text(encoding="utf-8"))
EVENTOS = pd.DataFrame(e["filas"], columns=e["cols"]).set_index("evento")
res = [procesar(eid) for eid in INCENDIOS]

# --- Curahuasi con sus resultados de los pasos 2, 5 y 6 (sus referencias también entrenan el modelo conjunto)
zc, uc = np.load(RAIZ / "data/processed/curahuasi/capas_curahuasi.npz"), np.load(RAIZ / "data/processed/curahuasi/uso_curahuasi.npz")
trc = Affine(*zc["transform"])
Xc = variables(np.load(RAIZ / "data/processed/curahuasi/ndvi_mensual_curahuasi.npz")["cubo"].astype("float32"), uc["elevacion"], trc)
cur = {"eid": "E41216", "nombre": "Curahuasi", "departamento": "Apurímac", "tr": trc, "H": zc["san"].shape[0], "W": zc["san"].shape[1],
       "quem": zc["sev"] > 0, "q027": (zc["sev"] >= 2), "sin_dato": zc["sin_dato"], "zona": zc["zona"], "san": zc["san"], "wc": zc["wc"],
       "el": uc["elevacion"], "X": Xc, "ref": referencias(zc["san"], zc["wc"], Xc, trc), "uso": uc["uso"],
       "antes": "2026-09-20", "despues": "2026-10-17", "valido_antes": 100, "valido_despues": 97}
todos = [cur] + res

# --- Random Forest conjunto: hasta 4 000 referencias por clase y recuadro; validación por bloques de 2 km
rng = np.random.default_rng(0)
Xs, ys, gs, fs = [], [], [], []
for i, r in enumerate(todos):
    for k in range(3):
        idx = np.flatnonzero(r["ref"] == k)
        idx = rng.choice(idx, min(4000, len(idx)), replace=False) if len(idx) else idx
        fil, col = np.divmod(idx, r["W"])
        Xs.append(r["X"][idx]); ys.append(np.full(len(idx), k)); gs.append(i * 10 ** 6 + (fil // 100) * 1000 + col // 100); fs.append(np.full(len(idx), i))
Xs, ys, gs, fs = map(np.concatenate, (Xs, ys, gs, fs))
print("referencias por recuadro y clase:", {r["nombre"]: [int((r["ref"] == k).sum()) for k in range(3)] for r in todos}, flush=True)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0)
pred = np.empty(len(ys), int)
for a, b in GroupKFold(5).split(Xs, ys, gs):
    rf.fit(Xs[a], ys[a]); pred[b] = rf.predict(Xs[b])
rep = classification_report(ys, pred, target_names=CLASES, digits=2, output_dict=True)
print(classification_report(ys, pred, target_names=CLASES, digits=2), flush=True)
rf.fit(Xs, ys)

# --- métricas por incendio y miniaturas a 40 m
filas, minis = [], []
for i, r in enumerate(todos):
    H, W, tr = r["H"], r["W"], r["tr"]
    if "uso" not in r:
        uso = np.zeros(H * W, "uint8")
        sel = r["san"].ravel() & np.isfinite(r["X"]).all(1)
        uso[sel] = rf.predict(r["X"][sel]) + 1
        r["uso"] = uso.reshape(H, W)
    uso, quem, san = r["uso"], r["quem"], r["san"]
    ha = abs(tr.a * tr.e) / 1e4
    dist = distance_transform_edt(~quem) * abs(tr.a) if quem.any() else np.full((H, W), np.inf)
    a1, a5 = san & ~quem & (dist <= 1000), san & ~quem & (dist <= 5000)
    bloque_ok = fs == i
    filas.append({
        "incendio": r["nombre"], "departamento": r["departamento"], "evento": r["eid"],
        "altitud_mediana_m": round(float(np.median(r["el"][quem]))) if quem.any() else None,
        "antes": r["antes"], "despues": r["despues"], "valido_antes": r["valido_antes"], "valido_despues": r["valido_despues"],
        "sin_dato_zona_pct": round(100 * float((r["zona"] & r["sin_dato"]).sum() / max(r["zona"].sum(), 1)), 1),
        "quemado_ha": round(float(quem.sum() * ha)), "quemado_027_ha": round(float(r["q027"].sum() * ha)),
        "agricola_quemada_ha": round(float((quem & san).sum() * ha)),
        "chacras_quemadas_ha": round(float((quem & (uso == 1)).sum() * ha)), "pasto_quemado_ha": round(float((quem & (uso == 2)).sum() * ha)),
        "arboles_quemados_ha": round(float((quem & (uso == 3)).sum() * ha)),
        "wc_cultivo_quemado_ha": round(float((quem & (r["wc"] == 40)).sum() * ha)),
        "agricola_1km_ha": round(float(a1.sum() * ha)), "chacras_1km_ha": round(float((a1 & (uso == 1)).sum() * ha)),
        "agricola_5km_ha": round(float(a5.sum() * ha)), "chacras_5km_ha": round(float((a5 & (uso == 1)).sum() * ha)),
        "recuadro_km": f"{W * abs(tr.a) / 1000:.1f} x {H * abs(tr.a) / 1000:.1f}",
        "exactitud_validacion_pct": round(100 * accuracy_score(ys[bloque_ok], pred[bloque_ok])) if bloque_ok.any() else None,
    })
    # códigos: 0 nada, 1 nube o sin dato, 2 agrícola sin quemar (pasto o árboles), 3 chacra sin quemar,
    #          4 quemado no agrícola, 5 quemado agrícola (pasto o árboles), 6 chacra quemada; bloques de 2 x 2 con el código mayor
    cod = np.zeros((H, W), "uint8")
    cod[r["sin_dato"]] = 1
    cod[san] = 2
    cod[san & (uso == 1)] = 3
    cod[quem] = 4
    cod[quem & san] = 5
    cod[quem & (uso == 1)] = 6
    h2, w2 = H // 2, W // 2
    c2 = cod[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).max(axis=(1, 3))
    minis.append({"incendio": r["nombre"], "ancho": w2, "alto": h2, "pixel_m": 2 * abs(tr.a),
                  "codigos": base64.b64encode(c2.tobytes()).decode()})

tabla = pd.DataFrame(filas)
tabla.to_csv(MON / "comparacion.csv", index=False)

# --- rasters para el visor web, en el mismo formato que Curahuasi (paso 3): dnbr, flags (zona, agua, SAN), wc y uso a 20 m;
#     color real antes y después a 40 m
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype="uint8").tobytes()).decode()
for r in res:
    slug = SLUGS[r["eid"]]
    tr = r["tr"]
    sin = r["sin_dato"] | ~np.isfinite(r["dnbr"])
    q = np.where(sin, 0, np.clip(np.floor((np.nan_to_num(r["dnbr"]) + 0.5) * 100 + 1e-6), 1, 255)).astype("uint8")
    flags = (r["zona"] * 1 | r["agua"] * 2 | r["san"] * 4).astype("uint8")
    filas_r = [{"fila": i, "dnbr": b64(q[i]), "flags": b64(flags[i]), "wc": b64(r["wc"][i]), "uso": b64(r["uso"][i])} for i in range(r["H"])]
    filas_r[0].update(ancho=r["W"], alto=r["H"], pixel_m=float(abs(tr.a)), x0=float(tr.c), y0=float(tr.f), crs=r["crs"])
    (MON / f"raster_{slug}.json").write_text(json.dumps({"rows": filas_r}), encoding="utf-8")
    ca, cd = np.moveaxis(r["rgb_antes"], 0, -1), np.moveaxis(r["rgb_despues"], 0, -1)
    (MON / f"color_{slug}.json").write_text(json.dumps({"rows": [{"fila": i, "antes": b64(ca[i]), "despues": b64(cd[i])} for i in range(ca.shape[0])]}), encoding="utf-8")
    np.savez_compressed(SAL / f"{r['eid']}_uso.npz", uso=r["uso"])
    print(f"   web: raster_{slug}.json {(MON / f'raster_{slug}.json').stat().st_size / 1e6:.1f} MB · color_{slug}.json {(MON / f'color_{slug}.json').stat().st_size / 1e6:.1f} MB", flush=True)
(MON / "miniaturas.json").write_text(json.dumps({"rows": minis}), encoding="utf-8")
val = {"exactitud": round(accuracy_score(ys, pred), 2), **{c: {m: round(rep[c][m], 2) for m in ("precision", "recall", "f1-score")} for c in CLASES}}
(SAL / "validacion_conjunta.json").write_text(json.dumps(val, indent=2, ensure_ascii=False), encoding="utf-8")
pd.set_option("display.width", 250)
print(tabla.drop(columns=["evento", "recuadro_km"]).to_string(index=False))
print(json.dumps(val, ensure_ascii=False))
