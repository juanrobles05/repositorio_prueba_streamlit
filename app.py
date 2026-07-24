"""
App de Streamlit: Análisis Exploratorio de Datos (EDA) de Tráfico Aéreo Sintético
-----------------------------------------------------------------------------
1. Genera datos sintéticos de vuelos (aerolíneas, rutas, retrasos, pasajeros, etc.)
2. Realiza EDA: análisis cuantitativo, cualitativo y gráficos interactivos.
3. Permite interacción del usuario mediante filtros, parámetros de generación
   y descarga de los datos generados.

Para ejecutar:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------
# Configuración general de la página
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="EDA Tráfico Aéreo Sintético",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ EDA de Tráfico Aéreo Sintético")
st.caption(
    "Genera datos sintéticos de vuelos y explóralos de forma interactiva: "
    "estadísticas cuantitativas, análisis cualitativo y visualizaciones."
)

# -----------------------------------------------------------------------
# 1. GENERACIÓN DE DATOS SINTÉTICOS
# -----------------------------------------------------------------------

AEROLINEAS = ["Avianca", "LATAM", "Wingo", "American Airlines", "Copa Airlines",
              "JetBlue", "Iberia", "Delta", "Aeroméxico", "Ryanair"]

CIUDADES = ["Bogotá", "Medellín", "Cali", "Miami", "Madrid", "Ciudad de México",
            "Nueva York", "Panamá", "São Paulo", "Lima", "Santiago", "Cancún"]

TIPOS_AVION = ["Airbus A320", "Boeing 737", "Airbus A330", "Boeing 787",
               "Embraer E190", "ATR 72"]

ESTADOS = ["A tiempo", "Retrasado", "Cancelado", "Desviado"]

DIST_ENTRE_CIUDADES_KM = {}  # se calcula de forma sintética a partir de coordenadas ficticias


def _generar_coordenadas_ciudades(seed: int) -> dict:
    """Asigna coordenadas ficticias fijas (pero reproducibles) a cada ciudad."""
    rng = np.random.default_rng(seed=123)  # fijo para que la distancia sea estable entre reruns
    coords = {}
    for ciudad in CIUDADES:
        lat = rng.uniform(-40, 40)
        lon = rng.uniform(-100, 10)
        coords[ciudad] = (lat, lon)
    return coords


def _distancia_km(origen: str, destino: str, coords: dict) -> float:
    lat1, lon1 = coords[origen]
    lat2, lon2 = coords[destino]
    # Aproximación simple tipo "distancia euclidiana" escalada a km (dato sintético, no geodésico real)
    dist = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111
    return max(dist, 150)  # distancia mínima razonable


@st.cache_data(show_spinner=False)
def generar_datos_sinteticos(n_vuelos: int, seed: int, prob_cancelado: float,
                              prob_desviado: float, fecha_inicio: str,
                              dias_rango: int) -> pd.DataFrame:
    """Genera un DataFrame sintético de vuelos con columnas cuantitativas y cualitativas."""
    rng = np.random.default_rng(seed)
    coords = _generar_coordenadas_ciudades(seed)

    fechas_base = pd.to_datetime(fecha_inicio) + pd.to_timedelta(
        rng.integers(0, dias_rango, size=n_vuelos), unit="D"
    )

    filas = []
    for i in range(n_vuelos):
        origen, destino = rng.choice(CIUDADES, size=2, replace=False)
        aerolinea = rng.choice(AEROLINEAS)
        avion = rng.choice(TIPOS_AVION)

        distancia = _distancia_km(origen, destino, coords)
        # Duración de vuelo sintética en función de la distancia + ruido
        duracion_min = distancia / 8.5 + rng.normal(0, 12)
        duracion_min = max(duracion_min, 35)

        hora_salida = rng.integers(0, 24)
        minuto_salida = rng.choice([0, 15, 30, 45])
        salida = fechas_base[i] + pd.Timedelta(hours=int(hora_salida), minutes=int(minuto_salida))

        # Probabilidad de estado del vuelo
        p = rng.random()
        if p < prob_cancelado:
            estado = "Cancelado"
            retraso = np.nan
            llegada = pd.NaT
        elif p < prob_cancelado + prob_desviado:
            estado = "Desviado"
            retraso = rng.integers(30, 180)
            llegada = salida + pd.Timedelta(minutes=int(duracion_min + retraso))
        else:
            # Retraso: la mayoría a tiempo o con poco retraso, cola larga ocasional
            retraso = max(0, rng.normal(8, 25))
            estado = "A tiempo" if retraso <= 15 else "Retrasado"
            llegada = salida + pd.Timedelta(minutes=int(duracion_min + retraso))

        pasajeros_capacidad = {
            "Airbus A320": 180, "Boeing 737": 175, "Airbus A330": 280,
            "Boeing 787": 250, "Embraer E190": 100, "ATR 72": 70,
        }[avion]
        factor_ocupacion = np.clip(rng.normal(0.78, 0.15), 0.35, 1.0)
        pasajeros = int(pasajeros_capacidad * factor_ocupacion)

        precio_ticket = round(max(50, distancia * rng.uniform(0.08, 0.18)), 2)

        filas.append({
            "vuelo_id": f"VL{i+1:05d}",
            "aerolinea": aerolinea,
            "origen": origen,
            "destino": destino,
            "tipo_avion": avion,
            "fecha_salida": salida,
            "fecha_llegada": llegada,
            "distancia_km": round(distancia, 1),
            "duracion_min": round(duracion_min, 1),
            "retraso_min": None if pd.isna(retraso) else round(float(retraso), 1),
            "estado": estado,
            "pasajeros": pasajeros,
            "capacidad": pasajeros_capacidad,
            "ocupacion_pct": round(factor_ocupacion * 100, 1),
            "precio_ticket_usd": precio_ticket,
        })

    df = pd.DataFrame(filas)
    df["dia_semana"] = df["fecha_salida"].dt.day_name()
    df["hora_salida"] = df["fecha_salida"].dt.hour
    df["mes"] = df["fecha_salida"].dt.month_name()
    return df


# -----------------------------------------------------------------------
# 2. BARRA LATERAL: PARÁMETROS INTERACTIVOS DE GENERACIÓN
# -----------------------------------------------------------------------
st.sidebar.header("⚙️ Parámetros de generación")

n_vuelos = st.sidebar.slider("Número de vuelos a generar", 100, 20000, 3000, step=100)
seed = st.sidebar.number_input("Semilla aleatoria (reproducibilidad)", value=42, step=1)
fecha_inicio = st.sidebar.date_input("Fecha de inicio", value=pd.to_datetime("2025-01-01"))
dias_rango = st.sidebar.slider("Rango de días a simular", 30, 365, 180)
prob_cancelado = st.sidebar.slider("Probabilidad de cancelación", 0.0, 0.2, 0.03, step=0.01)
prob_desviado = st.sidebar.slider("Probabilidad de desvío", 0.0, 0.2, 0.02, step=0.01)

if st.sidebar.button("🔄 Regenerar datos", use_container_width=True):
    st.cache_data.clear()

df = generar_datos_sinteticos(
    n_vuelos=n_vuelos,
    seed=int(seed),
    prob_cancelado=prob_cancelado,
    prob_desviado=prob_desviado,
    fecha_inicio=str(fecha_inicio),
    dias_rango=dias_rango,
)

# -----------------------------------------------------------------------
# 3. FILTROS INTERACTIVOS SOBRE LOS DATOS GENERADOS
# -----------------------------------------------------------------------
st.sidebar.header("🔍 Filtros de exploración")

aerolineas_sel = st.sidebar.multiselect("Aerolínea", sorted(df["aerolinea"].unique()))
origenes_sel = st.sidebar.multiselect("Origen", sorted(df["origen"].unique()))
estados_sel = st.sidebar.multiselect("Estado del vuelo", sorted(df["estado"].unique()))

df_filtrado = df.copy()
if aerolineas_sel:
    df_filtrado = df_filtrado[df_filtrado["aerolinea"].isin(aerolineas_sel)]
if origenes_sel:
    df_filtrado = df_filtrado[df_filtrado["origen"].isin(origenes_sel)]
if estados_sel:
    df_filtrado = df_filtrado[df_filtrado["estado"].isin(estados_sel)]

st.sidebar.markdown(f"**Registros tras filtros:** {len(df_filtrado):,}")

# -----------------------------------------------------------------------
# 4. PESTAÑAS: DATOS / EDA CUANTITATIVO / EDA CUALITATIVO / GRÁFICOS
# -----------------------------------------------------------------------
tab_datos, tab_cuanti, tab_cuali, tab_graficos = st.tabs(
    ["📄 Datos", "🔢 EDA Cuantitativo", "🔤 EDA Cualitativo", "📊 Gráficos"]
)

# --- Pestaña de datos ---
with tab_datos:
    st.subheader("Vista previa de los datos sintéticos")
    st.dataframe(df_filtrado.head(200), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de vuelos", f"{len(df_filtrado):,}")
    col2.metric("Aerolíneas distintas", df_filtrado["aerolinea"].nunique())
    col3.metric("Rutas distintas", (df_filtrado["origen"] + " → " + df_filtrado["destino"]).nunique())
    col4.metric("% Cancelados", f"{(df_filtrado['estado'].eq('Cancelado').mean()*100):.1f}%")

    csv = df_filtrado.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        data=csv,
        file_name="trafico_aereo_sintetico.csv",
        mime="text/csv",
    )

# --- Pestaña EDA cuantitativo ---
with tab_cuanti:
    st.subheader("Estadística descriptiva de variables numéricas")
    columnas_num = ["distancia_km", "duracion_min", "retraso_min", "pasajeros",
                     "ocupacion_pct", "precio_ticket_usd"]
    st.dataframe(df_filtrado[columnas_num].describe().T, use_container_width=True)

    st.subheader("Matriz de correlación")
    corr = df_filtrado[columnas_num].corr(numeric_only=True)
    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlación entre variables cuantitativas",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Selecciona una variable para ver su distribución numérica")
    var_num = st.selectbox("Variable numérica", columnas_num)
    st.write(df_filtrado[var_num].describe())

# --- Pestaña EDA cualitativo ---
with tab_cuali:
    st.subheader("Frecuencias de variables categóricas")
    columnas_cat = ["aerolinea", "origen", "destino", "tipo_avion", "estado", "dia_semana", "mes"]
    var_cat = st.selectbox("Variable categórica", columnas_cat)

    tabla_frecuencias = (
        df_filtrado[var_cat].value_counts()
        .rename_axis(var_cat)
        .reset_index(name="frecuencia")
    )
    tabla_frecuencias["porcentaje"] = round(
        tabla_frecuencias["frecuencia"] / tabla_frecuencias["frecuencia"].sum() * 100, 2
    )
    st.dataframe(tabla_frecuencias, use_container_width=True)

    st.subheader("Tabla cruzada (opcional)")
    col_a, col_b = st.columns(2)
    with col_a:
        var_cat_1 = st.selectbox("Variable 1", columnas_cat, index=0, key="cruce1")
    with col_b:
        var_cat_2 = st.selectbox("Variable 2", columnas_cat, index=4, key="cruce2")
    if var_cat_1 != var_cat_2:
        cruce = pd.crosstab(df_filtrado[var_cat_1], df_filtrado[var_cat_2])
        st.dataframe(cruce, use_container_width=True)
    else:
        st.info("Selecciona dos variables distintas para ver la tabla cruzada.")

# --- Pestaña de gráficos interactivos ---
with tab_graficos:
    st.subheader("Visualizaciones interactivas")

    st.markdown("**Distribución de vuelos por aerolínea**")
    fig1 = px.histogram(df_filtrado, x="aerolinea", color="estado",
                         title="Vuelos por aerolínea, coloreado por estado")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("**Distribución del retraso (minutos)**")
    fig2 = px.histogram(df_filtrado.dropna(subset=["retraso_min"]), x="retraso_min",
                         nbins=40, title="Distribución de retrasos")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Retraso promedio por hora de salida**")
    retraso_hora = df_filtrado.groupby("hora_salida", as_index=False)["retraso_min"].mean()
    fig3 = px.line(retraso_hora, x="hora_salida", y="retraso_min", markers=True,
                    title="Retraso promedio según hora de salida")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Ocupación vs. Precio del ticket**")
    fig4 = px.scatter(df_filtrado, x="ocupacion_pct", y="precio_ticket_usd",
                       color="aerolinea", size="pasajeros", opacity=0.6,
                       title="Relación entre ocupación y precio del ticket")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("**Boxplot de duración de vuelo por tipo de avión**")
    fig5 = px.box(df_filtrado, x="tipo_avion", y="duracion_min", color="tipo_avion",
                   title="Duración de vuelo por tipo de avión")
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("**Vuelos por día de la semana**")
    orden_dias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    conteo_dias = df_filtrado["dia_semana"].value_counts().reindex(orden_dias).reset_index()
    conteo_dias.columns = ["dia_semana", "conteo"]
    fig6 = px.bar(conteo_dias, x="dia_semana", y="conteo",
                  title="Número de vuelos por día de la semana")
    st.plotly_chart(fig6, use_container_width=True)

st.divider()
st.caption(
    "Datos 100% sintéticos generados con NumPy/Pandas para fines demostrativos "
    "de EDA. No representan tráfico aéreo real."
)
