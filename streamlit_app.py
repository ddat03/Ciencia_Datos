import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Configuración de página
st.set_page_config(page_title="Predictor de Reservorio", layout="wide")
st.title("🔍 Predictor de Calidad de Reservorio Petrolero")

# Sidebar para seleccionar modelo
st.sidebar.header("Configuración del Modelo")
st.sidebar.info("Modelo: RF - Class Weights (Original)")

# Cargar modelo
modelo_path = "rf_cw.pkl"

try:
    with open(modelo_path, 'rb') as f:
        modelo = pickle.load(f)
    st.sidebar.success(f"✓ Modelo cargado correctamente")
except FileNotFoundError:
    st.sidebar.error(f"❌ No se encontró: {modelo_path}")
    st.stop()

# Threshold recomendado
umbral_recomendado = 0.6

# División en columnas
col1, col2 = st.columns(2)

# Columna 1: Variables numéricas (entrada manual)
with col1:
    st.subheader("Variables Geofísicas")
    
    rmed = st.number_input("Resistividad Media (RMED) - Ohm-m", min_value=0.0, max_value=1000.0, value=50.0, step=1.0)
    cali = st.number_input("Calibre (CALI) - Pulgadas", min_value=6.0, max_value=20.0, value=12.0, step=0.1)
    sp = st.number_input("Potencial Espontáneo (SP) - mV", min_value=-200.0, max_value=200.0, value=0.0, step=5.0)
    dtc = st.number_input("Lentitud Compresional (DTC) - μs/ft", min_value=50.0, max_value=150.0, value=100.0, step=1.0)
    rxo = st.number_input("Resistividad de Zona Lavada (RXO) - Ohm-m", min_value=0.0, max_value=1000.0, value=30.0, step=1.0)
    pef = st.number_input("Factor Fotoeléctrico (PEF) - barns/e", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

# Columna 2: Variables categóricas y controles
with col2:
    st.subheader("Configuración Adicional")
    
    # Mapeo de profundidad a etiquetas legibles
    formacion_por_profundidad = {
        0: (0, 500),
        1: (500, 800),
        2: (800, 1200),
        3: (1200, 1600),
        4: (1600, 2000),
        5: (2000, 2400),
        6: (2400, 2800),
        7: (2800, 3200),
        8: (3200, 3600),
        9: (3600, 4000),
        10: (4000, 4400),
        11: (4400, 4800),
        12: (4800, 5200),
        13: (5200, 5600),
        14: (5600, 6000),
        15: (6000, 6400),
        16: (6400, 6800),
        17: (6800, 7200),
        18: (7200, 7600),
        19: (7600, 8000),
        20: (8000, 8500)
    }
    
    # Crear opciones legibles para el usuario
    formation_display = {f"{rango[0]}-{rango[1]}m": idx for idx, rango in formacion_por_profundidad.items()}
    formation_str = st.selectbox("Rango de Profundidad (metros)", list(formation_display.keys()))
    formation = formation_display[formation_str]
    
    st.subheader("Ajustes de Predicción")
    threshold = st.slider("Threshold de Decisión", min_value=0.0, max_value=1.0, value=umbral_recomendado, step=0.05)
    st.info(f"Threshold recomendado: {umbral_recomendado}")
    
    # Botón de predicción
    if st.button("Predecir", use_container_width=True):
        # Preparar datos en el ORDEN EXACTO que espera el modelo
        df_entrada = pd.DataFrame({
            'FORMATION': [formation],
            'Calibre (CALI)': [cali],
            'Resistividad Media (RMED)': [rmed],
            'Factor Fotoeléctrico (PEF)': [pef],
            'Lentitud Compresional (DTC)': [dtc],
            'Potencial Espontáneo (SP)': [sp],
            'Resistividad de Zona Lavada (RXO)': [rxo]
        })
        
        # Reordenar columnas en el orden exacto que espera el modelo
        columnas_ordenadas = ['FORMATION', 'Calibre (CALI)', 'Resistividad Media (RMED)', 
                             'Factor Fotoeléctrico (PEF)', 'Lentitud Compresional (DTC)', 
                             'Potencial Espontáneo (SP)', 'Resistividad de Zona Lavada (RXO)']
        df_entrada = df_entrada[columnas_ordenadas]
        
        # Predicción
        probabilidad = modelo.predict_proba(df_entrada)[0][1]
        prediccion = (probabilidad >= threshold).astype(int)
        
        # Resultados
        st.divider()
        st.subheader("Resultados")
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            if prediccion == 1:
                st.success("✓ RESERVORIO DETECTADO")
            else:
                st.error("✗ NO RESERVORIO")
            st.metric("Probabilidad", f"{probabilidad:.2%}")
        
        with col_r2:
            st.metric("Confianza", f"{abs(probabilidad - 0.5) * 2:.2%}")
            st.metric("Threshold", f"{threshold:.2f}")
        
        # Tabla
        resumen = pd.DataFrame({
            'Variable': ['Resistividad Media (RMED)', 'Calibre (CALI)', 'Potencial Espontáneo (SP)', 'Lentitud Compresional (DTC)', 'Resistividad de Zona Lavada (RXO)', 'Factor Fotoeléctrico (PEF)', 'Rango de Profundidad'],
            'Valor': [rmed, cali, sp, dtc, rxo, pef, formation_str]
        })
        st.table(resumen)

st.divider()
st.caption("Predictor de Reservorio v1.0 | Fase 2 Midterm")
