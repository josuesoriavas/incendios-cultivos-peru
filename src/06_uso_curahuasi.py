"""
Paso 6: ¿qué hay realmente dentro de la superficie agrícola de Curahuasi? Cultivo, pasto o matorral, o árboles.

La regla de Dinámica Agrícola (paso 5) se diseñó para la costa, donde lo que no es cultivo es desierto. En el valle
interandino el pasto y el matorral tampoco se secan del todo, y la regla los llama "permanentes". Aquí se entrena un
Random Forest con el NDVI máximo de cada uno de los 12 meses previos al fuego, su mínimo, su máximo y la altitud,
usando como referencia los píxeles donde dos fuentes coinciden:
  cultivo           superficie agrícola del MIDAGRI que WorldCover 2021 también llama Cultivo
  pasto o matorral  fuera de la superficie agrícola (a más de 60 m), WorldCover Pastizal o Matorral
  árboles           fuera de la superficie agrícola (a más de 60 m), WorldCover Bosque
Validación cruzada espacial: 5 pliegues por bloques de 2 km. Luego se clasifica cada píxel de la superficie agrícola.
Salidas: data/processed/curahuasi/uso_curahuasi.npz y uso_resumen.json
Uso:  uv run python src/06_uso_curahuasi.py
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt, uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupKFold

RAIZ = Path(__file__).resolve().parent.parent
SAL = RAIZ / "data/processed/curahuasi"
DEM = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_S14_00_W073_00_DEM/Copernicus_DSM_COG_10_S14_00_W073_00_DEM.tif"
CLASES = ["cultivo", "pasto o matorral", "árboles"]
z = np.load(SAL / "capas_curahuasi.npz")
H, W = z["dnbr"].shape
tr = Affine(*z["transform"])
ha = abs(tr.a * tr.e) / 1e4

# --- altitud: Copernicus DEM de 30 m llevado a la grilla de 20 m
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open(DEM) as s:
    w = from_bounds(-72.62, -13.76, -72.31, -13.47, s.transform).round_offsets().round_lengths()
    el = np.zeros((H, W), "float32")
    reproject(s.read(1, window=w).astype("float32"), el, src_transform=s.window_transform(w), src_crs=s.crs,
              dst_transform=tr, dst_crs=str(z["crs"]), resampling=Resampling.bilinear)

# --- variables por píxel: 12 meses (los vacíos con la mediana del píxel), mínimo, máximo y altitud
cubo = np.load(SAL / "ndvi_mensual_curahuasi.npz")["cubo"].astype("float32")
with np.errstate(all="ignore"):
    med = np.nanmedian(cubo, 0)
cubo = np.where(np.isfinite(cubo), cubo, med)
gy, gx = np.gradient(el, abs(tr.a))
pendiente = np.degrees(np.arctan(np.hypot(gx, gy)))
# meses: 0 = oct 2025 ... 11 = sep 2026; lluvias = ene-mar (3-5), secas = jun-ago (8-10)
X = np.column_stack([cubo.reshape(12, -1).T, cubo.min(0).ravel(), cubo.max(0).ravel(), (cubo.max(0) - cubo.min(0)).ravel(),
                     cubo.argmax(0).ravel(), cubo[3:6].mean(0).ravel(), cubo[8:11].mean(0).ravel(), el.ravel(), pendiente.ravel()])
valido = np.isfinite(X).all(1)

# --- referencias donde dos fuentes coinciden; solo píxeles puros (su entorno de 3 x 3 es de la misma clase)
puro = lambda m: uniform_filter(m.astype("float32"), 3).ravel() > 0.999
san, wc = z["san"].ravel(), z["wc"].ravel()
lejos = (distance_transform_edt(~z["san"]) * abs(tr.a)).ravel() > 200  # pasto natural, lejos de la superficie agrícola
wc2 = z["wc"]
ref = np.full(H * W, -1)
ref[puro(z["san"] & (wc2 == 40))] = 0
ref[~san & lejos & puro(np.isin(wc2, [20, 30]))] = 1
ref[~san & lejos & puro(wc2 == 10)] = 2
ref[~valido] = -1
print("píxeles de referencia:", {c: int((ref == k).sum()) for k, c in enumerate(CLASES)}, flush=True)

rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.flatnonzero(ref == k), min(15000, int((ref == k).sum())), replace=False) for k in range(3)])
fil, col = np.divmod(idx, W)
bloque = (fil // 100) * 1000 + col // 100  # bloques de 2 km (100 píxeles de 20 m)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0)
pred = np.empty(len(idx), int)
for a, b in GroupKFold(5).split(idx, ref[idx], bloque):
    rf.fit(X[idx[a]], ref[idx[a]])
    pred[b] = rf.predict(X[idx[b]])
rep = classification_report(ref[idx], pred, target_names=CLASES, digits=2, output_dict=True)
print(classification_report(ref[idx], pred, target_names=CLASES, digits=2), flush=True)

# --- modelo final y clasificación de la superficie agrícola: 0 fuera o sin dato, 1 cultivo, 2 pasto o matorral, 3 árboles
rf.fit(X[idx], ref[idx])
uso = np.zeros(H * W, "uint8")
sel = san & valido
uso[sel] = rf.predict(X[sel]) + 1
uso = uso.reshape(H, W)
np.savez_compressed(SAL / "uso_curahuasi.npz", uso=uso, elevacion=el)

# --- cruce con lo quemado y con la distancia a la cicatriz
quem, sanm = z["sev"] > 0, z["san"]
d = distance_transform_edt(~quem) * abs(tr.a)
din = np.load(SAL / "dinamica_curahuasi.npz")["din"]
grupos = {"agricola_en_el_recuadro": sanm, "agricola_quemada": sanm & quem,
          "agricola_sin_quemar_a_1km": sanm & ~quem & (d <= 1000), "agricola_sin_quemar_1_a_5km": sanm & ~quem & (d > 1000) & (d <= 5000)}
res = {
    "referencias": {c: int((ref == k).sum()) for k, c in enumerate(CLASES)},
    "validacion_espacial": {"exactitud": round(accuracy_score(ref[idx], pred), 2),
                            **{c: {m: round(rep[c][m], 2) for m in ("precision", "recall", "f1-score")} for c in CLASES}},
    **{g: {c: round(float((m & (uso == k + 1)).sum() * ha)) for k, c in enumerate(CLASES)} for g, m in grupos.items()},
    "quemada_regla_costa_vs_modelo": {f"{['sin dato', 'barbecho', 'transitorio', 'permanente'][a]} -> {CLASES[b - 1]}":
                                      round(float((sanm & quem & (din == a) & (uso == b)).sum() * ha)) for a in range(4) for b in (1, 2, 3)
                                      if (sanm & quem & (din == a) & (uso == b)).any()},
}
(SAL / "uso_resumen.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=2, ensure_ascii=False))
