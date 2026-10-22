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
c[0].metric("Focos en cultivo según WorldCover (monitor)", f"{r1['pct_focos_cultivo_worldcover_monitor']} %")
c[1].metric("Focos en cultivo según SAN 2024 (MIDAGRI)", f"{r1['pct_focos_en_cultivo_san']} %")
c[2].metric("Cultivo a ≤ 5 km de incendios activos", f"{r1['cultivo_a_5km_activos_ha']:,} ha")
c[3].metric("Cultivo quemado en Curahuasi (SAN 2024)", f"{r2['san2024_en_quemado_ha']:,} ha")

st.subheader("Curahuasi: área quemada (dNBR) sobre la superficie agrícola")
a, b = st.columns([3, 2])
a.image(str(R / "outputs/maps/curahuasi_dnbr_cultivo.png"))
b.markdown(f"""
- **Imágenes:** antes {r2['antes']} ({r2['valido_zona_antes_pct']} % válido en la zona), después {r2['despues']} ({r2['valido_zona_despues_pct']} % válido).
- **Quemado:** {r2['quemado_dnbr_010_ha']:,} ha con dNBR ≥ 0,10 · {r2['quemado_dnbr_027_ha']:,} ha con dNBR ≥ 0,27 · RBR ≥ 0,10: {r2['quemado_rbr_010_ha']:,} ha.
- **Monitor del profesor:** {r2['monitor_profesor_ha']:,.0f} ha ({r2['monitor_profesor_fechas']['pre']} → {r2['monitor_profesor_fechas']['post']}).
- **Sin dato (nubes) en la zona:** {r2['pct_sin_dato_en_zona']} % · agua excluida: {r2['agua_excluida_ha']} ha.
""")
b.write("**WorldCover dentro del área quemada (ha)**"); b.bar_chart(pd.Series(r2["worldcover_en_quemado_ha"]))
b.write("**Cultivo SAN 2024 quemado, según cómo lo ve WorldCover (ha)**"); b.bar_chart(pd.Series(r2["san2024_en_quemado_por_worldcover_ha"]))

st.subheader("¿Qué se puede quemar? Incendios activos con más cultivo alrededor")
act = inc[inc.activo].sort_values("cultivo_5km_ha", ascending=False)
m = folium.Map(location=[-10, -75], zoom_start=5, tiles="CartoDB positron")
for _, x in act.iterrows():
    folium.CircleMarker([x.lat, x.lon], radius=3 + min(x.cultivo_5km_ha, 5000) / 400, color="#b4441a", fill=True, fill_opacity=0.6,
                        tooltip=f"{x.distrito} ({x.departamento}) · huella {x.huella_ha:.0f} ha · cultivo a 5 km {x.cultivo_5km_ha:.0f} ha").add_to(m)
izq, der = st.columns([3, 2])
with izq:
    st_folium(m, height=520, use_container_width=True)
der.dataframe(act[["distrito", "departamento", "huella_ha", "cultivo_en_huella_ha", "cultivo_1km_ha", "cultivo_5km_ha", "prof_cobertura"]]
              .head(20).round(0), hide_index=True, height=520)
st.caption("Huella = píxeles VIIRS de 375 m con fuego; no es área quemada. Un foco de calor no es un incendio. Las nubes esconden el fuego. "
           "SAN 2024 es superficie agrícola (incluye barbecho), no cultivo en pie.")
