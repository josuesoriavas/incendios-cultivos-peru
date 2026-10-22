# Cultivos en riesgo de quema — Perú, octubre 2026

Tarea de Ciencia de Datos con Python (Prof. Alexander Quispe, Universidad del Pacífico) · retos **C (¿qué se quemó?)** y **F (clima y riesgo)**.

## Pregunta

1. **¿Qué se quemó?** Del área quemada en incendios activos de octubre de 2026, ¿cuántas hectáreas eran cultivo, pastizal o bosque?
2. **¿Qué se puede quemar?** ¿Qué zonas de cultivo están cerca de incendios activos y con clima favorable a que el fuego crezca esta semana (baja humedad, viento fuerte, sin lluvia)?

## Datos (fechas y fuentes)

| Dato | Fuente | Tipo |
|---|---|---|
| Focos de calor (últimos 7 días) | NASA FIRMS, VIIRS «South America 7d» | Puntos |
| Imágenes antes/después | Sentinel-2 L2A, Microsoft Planetary Computer | Raster |
| Cobertura del suelo | ESA WorldCover 2021, 10 m | Raster |
| Clima diario y pronóstico | Open-Meteo (sin clave) | Serie |
| Límites distritales | juaneladio/peru-geojson | Polígonos |

_Pendiente: fechas exactas de las imágenes y % de nubes._

## Método

_Pendiente._

## Hallazgos

_Pendiente: 3 hallazgos con números._

## Mapa

_Pendiente._

## Limitaciones

- Un foco de calor no es un incendio.
- Las nubes esconden el fuego y alteran el dNBR.
- WorldCover es de 2021: la cobertura pudo cambiar.

## Estructura

```
data/raw         datos descargados (no se suben a GitHub)
data/processed   tablas y capas procesadas
src              scripts de Python
notebooks        exploración
outputs/maps     mapas exportados
dashboard        tablero (Streamlit)
```
