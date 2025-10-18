import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Configuración de página
st.set_page_config(page_title="Predictor de Reservorio", layout="wide")
st.title("🔍 Predictor de Calidad de Reservorio Petrolero")

# Sidebar para seleccionar modelo
st.sidebar.header("Configuración del Modelo")
modelo_seleccionado = st.sidebar.selectbox(
    "Selecciona el modelo",
    ["RF - Class Weights (Original)", "RF - Optimizado"]
)

# Mapeo de modelos
modelos = {
    "RF - Class Weights (Original)": "mejor_modelo_rf_cw.pkl",
    "RF - Optimizado": "modelo_rf_optimizado.pkl"
}

# Cargar modelo
try:
    with open(modelos[modelo_seleccionado], 'rb') as f:
        modelo = pickle.load(f)
    st.sidebar.success(f"✓ Modelo cargado: {modelo_seleccionado}")
except FileNotFoundError:
    st.sidebar.error(f"❌ No se encontró el archivo: {modelos[modelo_seleccionado]}")
    st.stop()

# Threshold según modelo
if "Original" in modelo_seleccionado:
    threshold_default = 0.5
    umbral_recomendado = 0.6
else:
    threshold_default = 0.5
    umbral_recomendado = 0.5

# División en columnas
col1, col2 = st.columns(2)

# Columna 1: Variables numéricas
with col1:
    st.subheader("Variables Geofísicas")
    
    rmed = st.number_input(
        "Resistividad Media (RMED) - Ohm-m",
        min_value=0.0,
        max_value=1000.0,
        value=50.0,
        step=1.0,
        help="Rango típico: 0-1000 Ohm-m"
    )
    
    cali = st.number_input(
        "Calibre (CALI) - Pulgadas",
        min_value=6.0,
        max_value=20.0,
        value=12.0,
        step=0.1,
        help="Rango típico: 6-20 pulgadas"
    )
    
    sp = st.number_input(
        "Potencial Espontáneo (SP) - mV",
        min_value=-200.0,
        max_value=200.0,
        value=0.0,
        step=5.0,
        help="Rango típico: -200 a +200 mV"
    )
    
    dtc = st.number_input(
        "Lentitud Compresional (DTC) - μs/ft",
        min_value=50.0,
        max_value=150.0,
        value=100.0,
        step=1.0,
        help="Rango típico: 50-150 μs/ft"
    )
    
    rxo = st.number_input(
        "Resistividad de Zona Lavada (RXO) - Ohm-m",
        min_value=0.0,
        max_value=1000.0,
        value=30.0,
        step=1.0,
        help="Rango típico: 0-1000 Ohm-m"
    )
    
    pef = st.number_input(
        "Factor Fotoeléctrico (PEF) - barns/e",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
        help="Rango típico: 0-10 barns/e"
    )

# Columna 2: Variables categóricas y controles
with col2:
    st.subheader("Configuración Adicional")
    
    formation_options = {
        "Arenisca": 0,
        "Lutita": 1,
        "Caliza": 2,
        "Dolomita": 3,
        "Otra": 4
    }
    
    formation_str = st.selectbox(
        "Formación Geológica",
        list(formation_options.keys()),
        help="Selecciona la formación geológica del intervalo"
    )
    formation = formation_options[formation_str]
    
    st.subheader("Ajustes de Predicción")
    
    threshold = st.slider(
        "Threshold de Decisión",
        min_value=0.0,
        max_value=1.0,
        value=umbral_recomendado,
        step=0.05,
        help="Probabilidad mínima para clasificar como RESERVORIO"
    )
    
    st.info(f"📌 Threshold recomendado: {umbral_recomendado}")
    
    # Botón de predicción
    if st.button("🔮 Predecir", use_container_width=True):
        # Preparar datos
        datos = {
            'CALI': [cali],
            'RMED': [rmed],
            'DTC': [dtc],
            'SP': [sp],
            'RXO': [rxo],
            'PEF': [pef],
            'FORMATION': [formation]
        }
        
        df_entrada = pd.DataFrame(datos)
        
        # Realizar predicción
        probabilidad = modelo.predict_proba(df_entrada)[0][1]
        prediccion = (probabilidad >= threshold).astype(int)
        
        # Mostrar resultados
        st.divider()
        st.subheader("📊 Resultados de Predicción")
        
        col_resultado1, col_resultado2 = st.columns(2)
        
        with col_resultado1:
            if prediccion == 1:
                st.success("✅ RESERVORIO DETECTADO")
                st.metric("Probabilidad", f"{probabilidad:.2%}")
            else:
                st.error("❌ NO RESERVORIO")
                st.metric("Probabilidad", f"{probabilidad:.2%}")
        
        with col_resultado2:
            st.metric("Confianza", f"{abs(probabilidad - 0.5) * 2:.2%}")
            st.metric("Threshold Aplicado", f"{threshold:.2f}")
        
        # Tabla de resumen
        st.subheader("📋 Resumen de Inputs")
        resumen = pd.DataFrame({
            'Variable': ['RMED', 'CALI', 'SP', 'DTC', 'RXO', 'PEF', 'FORMATION'],
            'Valor': [rmed, cali, sp, dtc, rxo, pef, formation_str]
        })
        st.table(resumen)
        
        # Información de interpretación
        st.divider()
        st.subheader("ℹ️ Interpretación")
        
        if probabilidad >= threshold:
            st.write(f"""
            **Resultado:** Se predice presencia de RESERVORIO con {probabilidad:.1%} de confianza.
            
            **Recomendación:** Considerar perforación exploratoria en este intervalo.
            """)
        else:
            st.write(f"""
            **Resultado:** NO se predice presencia de reservorio ({probabilidad:.1%} de probabilidad).
            
            **Recomendación:** Evaluar otros intervalos de profundidad.
            """)

# Footer
st.divider()
st.caption("🔬 Predictor de Reservorio v1.0 | Fase 2 Midterm")