"""Tablero: cultivos e incendios en el Perú, octubre 2026.  Uso:  uv run streamlit run dashboard/app.py"""
import json
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

R = Path(__file__).resolve().parent.parent
P = R / "data/processed"
st.set_page_config(page_title="Cultivos e incendios · Perú 2026", layout="wide")
r1 = json.loads((P / "resumen_paso1.json").read_text(encoding="utf-8"))
r2 = json.loads((P / "curahuasi/resumen_curahuasi.json").read_text(encoding="utf-8"))
inc = pd.read_csv(P / "incendios.csv", encoding="utf-8-sig")

st.title("¿Qué se quemó y qué se puede quemar? Cultivos frente a los incendios de octubre 2026")
st.caption(f"Focos VIIRS (NASA FIRMS) del {r1['desde_utc']} al {r1['corte_utc']} UTC · Superficie Agrícola Nacional 2024 (MIDAGRI) · "
           f"Sentinel-2 L2A (Earth Search) · ESA WorldCover 2021")

c = st.columns(4)
c[0].metric("Focos en cultivo: WorldCover", f"{r1['pct_focos_cultivo_worldcover_monitor']} %", help="ESA WorldCover 2021, focos de 7 días al 17 oct")
c[1].metric("Focos en cultivo: MIDAGRI", f"{r1['pct_focos_en_cultivo_san']} %", help="Superficie Agrícola Nacional 2024 (MIDAGRI)")
c[2].metric("Agrícola a ≤ 5 km del fuego", f"{r1['cultivo_a_5km_activos_ha']:,} ha", help="Alrededor de los incendios activos (focos en 24 h)")
c[3].metric("Agrícola quemada, Curahuasi", f"{r2['san2024_en_quemado_ha']:,} ha", help="SAN 2024 dentro del área con dNBR ≥ 0,10")

st.subheader("Curahuasi: área quemada (dNBR) sobre la superficie agrícola")
a, b = st.columns([3, 2])
a.image(str(R / "outputs/maps/curahuasi_dnbr_cultivo.png"))
b.markdown(f"""
- **Imágenes:** antes {r2['antes']} ({r2['valido_zona_antes_pct']} % válido en la zona), después {r2['despues']} ({r2['valido_zona_despues_pct']} % válido).
- **Quemado:** {r2['quemado_dnbr_010_ha']:,} ha con dNBR ≥ 0,10 · {r2['quemado_dnbr_027_ha']:,} ha con dNBR ≥ 0,27 · RBR ≥ 0,10: {r2['quemado_rbr_010_ha']:,} ha.
- **Medición independiente:** 1,786 ha con dNBR ≥ 0,27 (imágenes del 5 y 17 oct).
- **Sin dato (nubes) en la zona:** {r2['pct_sin_dato_en_zona']} % · agua excluida: {r2['agua_excluida_ha']} ha.
""")
serie = lambda d: pd.Series({k: v for k, v in d.items() if v > 0 and k != "0"}).sort_values(ascending=False)
b.write("**WorldCover dentro del área quemada (ha)**"); b.bar_chart(serie(r2["worldcover_en_quemado_ha"]), horizontal=True)
b.write("**Superficie agrícola (SAN 2024) quemada, según cómo la clasifica WorldCover (ha)**")
b.bar_chart(serie(r2["san2024_en_quemado_por_worldcover_ha"]), horizontal=True, color="#5b8c3a")

st.subheader("¿Qué se puede quemar? Incendios activos con más cultivo alrededor")
act = inc[inc.activo].sort_values("cultivo_5km_ha", ascending=False)
m = folium.Map(location=[-9.5, -75], zoom_start=5, tiles="OpenStreetMap")
for _, x in act.iterrows():
    folium.CircleMarker([x.lat, x.lon], radius=2 + (x.cultivo_5km_ha ** 0.5) / 12, color="#b4441a", weight=1, fill=True, fill_opacity=0.5,
                        tooltip=f"{x.distrito} ({x.departamento}) · huella {x.huella_ha:.0f} ha · cultivo a 5 km {x.cultivo_5km_ha:.0f} ha").add_to(m)
izq, der = st.columns([3, 2])
with izq:
    st_folium(m, height=520, use_container_width=True)
der.dataframe(act[["distrito", "departamento", "cultivo_5km_ha", "cultivo_1km_ha", "huella_ha", "prof_cobertura"]]
              .head(20).round(0).rename(columns={"cultivo_5km_ha": "agrícola 5 km (ha)", "cultivo_1km_ha": "agrícola 1 km (ha)",
                                                 "huella_ha": "huella (ha)", "prof_cobertura": "WorldCover"}),
              hide_index=True, height=520)
st.caption("Huella = píxeles VIIRS de 375 m con fuego; no es área quemada. Un foco de calor no es un incendio. Las nubes esconden el fuego. "
           "SAN 2024 es superficie agrícola (incluye barbecho), no cultivo en pie.")
