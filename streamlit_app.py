import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Configuración de página
st.set_page_config(page_title="Predictor de Reservorio", layout="wide")
st.title("🔍 Predictor de Calidad de Reservorio Petrolero")

# Cargar dataset
@st.cache_data
def cargar_dataset():
    try:
        df = pd.read_csv('dataset_limpio.csv')
        return df
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'dataset_limpio.csv'")
        return None

dataset = cargar_dataset()

# Sidebar para seleccionar modelo
st.sidebar.header("Configuración del Modelo")
modelo_seleccionado = st.sidebar.selectbox(
    "Selecciona el modelo",
    ["RF - Class Weights (Original)", "RF - Optimizado"]
)

# Mapeo de modelos
modelos = {
    "RF - Class Weights (Original)": "rf_cw.pkl",
}

# Cargar modelo
try:
    with open(modelos[modelo_seleccionado], 'rb') as f:
        modelo = pickle.load(f)
    st.sidebar.success(f"✓ Modelo cargado: {modelo_seleccionado}")
except FileNotFoundError:
    st.sidebar.error(f"❌ No se encontró: {modelos[modelo_seleccionado]}")
    st.stop()

# Threshold según modelo
if "Original" in modelo_seleccionado:
    umbral_recomendado = 0.6
else:
    umbral_recomendado = 0.5

# Modo de entrada
st.sidebar.header("Modo de Entrada")
modo_entrada = st.sidebar.radio(
    "¿Cómo deseas ingresar datos?",
    ["Manual", "Desde Dataset", "Comparar Predicciones"]
)

# División en columnas
col1, col2 = st.columns(2)

if modo_entrada == "Manual":
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
        
        formation_options = {"Arenisca": 0, "Lutita": 1, "Caliza": 2, "Dolomita": 3, "Otra": 4}
        formation_str = st.selectbox("Formación Geológica", list(formation_options.keys()))
        formation = formation_options[formation_str]
        
        st.subheader("Ajustes de Predicción")
        threshold = st.slider("Threshold de Decisión", min_value=0.0, max_value=1.0, value=umbral_recomendado, step=0.05)
        st.info(f"Threshold recomendado: {umbral_recomendado}")
        
        # Botón de predicción
        if st.button("Predecir", use_container_width=True, key="btn_manual"):
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
                'Variable': ['RMED', 'CALI', 'SP', 'DTC', 'RXO', 'PEF', 'FORMATION'],
                'Valor': [rmed, cali, sp, dtc, rxo, pef, formation_str]
            })
            st.table(resumen)

elif modo_entrada == "Desde Dataset":
    if dataset is not None:
        st.subheader("📊 Seleccionar Muestra del Dataset")
        
        # Mostrar estadísticas del dataset
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("Total de muestras", len(dataset))
        with col_info2:
            st.metric("Columnas disponibles", len(dataset.columns))
        
        # Filtros opcionales
        with col1:
            st.subheader("Filtros (Opcional)")
            filtrar = st.checkbox("Aplicar filtros")
            
            datos_filtrados = dataset.copy()
            
            if filtrar:
                if 'RMED' in dataset.columns:
                    rmed_min, rmed_max = st.slider("Rango RMED", 
                                                     float(dataset['RMED'].min()), 
                                                     float(dataset['RMED'].max()),
                                                     (float(dataset['RMED'].min()), float(dataset['RMED'].max())))
                    datos_filtrados = datos_filtrados[(datos_filtrados['RMED'] >= rmed_min) & (datos_filtrados['RMED'] <= rmed_max)]
                
                if 'FORMATION' in dataset.columns:
                    formaciones = st.multiselect("Formaciones", dataset['FORMATION'].unique(), 
                                                 default=list(dataset['FORMATION'].unique()))
                    datos_filtrados = datos_filtrados[datos_filtrados['FORMATION'].isin(formaciones)]
        
        # Seleccionar muestra
        with col2:
            st.subheader("Seleccionar Muestra")
            idx_muestra = st.selectbox("Elige una muestra", range(len(datos_filtrados)), 
                                       format_func=lambda x: f"Muestra {x}")
            
            # Información de la muestra seleccionada
            muestra = datos_filtrados.iloc[idx_muestra]
            
            st.subheader("Datos de la Muestra Seleccionada")
            st.dataframe(muestra.to_frame().T, use_container_width=True)
        
        # Predicción con datos del dataset
        st.subheader("Ajustes de Predicción")
        threshold = st.slider("Threshold de Decisión", min_value=0.0, max_value=1.0, 
                             value=umbral_recomendado, step=0.05, key="threshold_dataset")
        
        if st.button("Predecir con Datos del Dataset", use_container_width=True, key="btn_dataset"):
            # Preparar features esperadas por el modelo
            features_requeridas = ['CALI', 'RMED', 'DTC', 'SP', 'RXO', 'PEF', 'FORMATION']
            
            # Verificar que existan todas las columnas
            columnas_faltantes = [col for col in features_requeridas if col not in muestra.index]
            
            if columnas_faltantes:
                st.error(f"❌ Faltan columnas en el dataset: {columnas_faltantes}")
            else:
                # Preparar datos en el orden correcto
                df_entrada = pd.DataFrame({
                    'CALI': [muestra['CALI']],
                    'RMED': [muestra['RMED']],
                    'DTC': [muestra['DTC']],
                    'SP': [muestra['SP']],
                    'RXO': [muestra['RXO']],
                    'PEF': [muestra['PEF']],
                    'FORMATION': [muestra['FORMATION']]
                })
                
                # Predicción
                probabilidad = modelo.predict_proba(df_entrada)[0][1]
                prediccion = (probabilidad >= threshold).astype(int)
                
                # Resultados
                st.divider()
                st.subheader("Resultados de la Predicción")
                
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

elif modo_entrada == "Comparar Predicciones":
    if dataset is not None:
        st.subheader("📈 Comparar Múltiples Predicciones")
        
        # Seleccionar cantidad de muestras a comparar
        cantidad = st.slider("¿Cuántas muestras deseas comparar?", 1, min(20, len(dataset)), 5)
        
        # Threshold
        threshold = st.slider("Threshold de Decisión", min_value=0.0, max_value=1.0, 
                             value=umbral_recomendado, step=0.05, key="threshold_compare")
        
        if st.button("Generar Predicciones Comparativas", use_container_width=True):
            # Seleccionar muestras aleatorias
            muestras_indices = np.random.choice(len(dataset), size=cantidad, replace=False)
            
            features_requeridas = ['CALI', 'RMED', 'DTC', 'SP', 'RXO', 'PEF', 'FORMATION']
            
            # Verificar que existan todas las columnas
            columnas_faltantes = [col for col in features_requeridas if col not in dataset.columns]
            
            if columnas_faltantes:
                st.error(f"❌ Faltan columnas en el dataset: {columnas_faltantes}")
            else:
                resultados = []
                
                for idx in muestras_indices:
                    muestra = dataset.iloc[idx]
                    
                    df_entrada = pd.DataFrame({
                        'CALI': [muestra['CALI']],
                        'RMED': [muestra['RMED']],
                        'DTC': [muestra['DTC']],
                        'SP': [muestra['SP']],
                        'RXO': [muestra['RXO']],
                        'PEF': [muestra['PEF']],
                        'FORMATION': [muestra['FORMATION']]
                    })
                    
                    probabilidad = modelo.predict_proba(df_entrada)[0][1]
                    prediccion = (probabilidad >= threshold).astype(int)
                    
                    resultados.append({
                        'Índice': idx,
                        'RMED': muestra['RMED'],
                        'CALI': muestra['CALI'],
                        'SP': muestra['SP'],
                        'DTC': muestra['DTC'],
                        'Probabilidad': f"{probabilidad:.2%}",
                        'Predicción': "✓ Reservorio" if prediccion == 1 else "✗ No Reservorio",
                        'Confianza': f"{abs(probabilidad - 0.5) * 2:.2%}"
                    })
                
                df_resultados = pd.DataFrame(resultados)
                st.dataframe(df_resultados, use_container_width=True)
                
                # Estadísticas
                st.divider()
                st.subheader("Estadísticas de Predicciones")
                
                col_est1, col_est2, col_est3 = st.columns(3)
                
                reservorios = sum(1 for r in resultados if "✓" in r['Predicción'])
                
                with col_est1:
                    st.metric("Muestras Analizadas", len(resultados))
                with col_est2:
                    st.metric("Reservorios Detectados", reservorios)
                with col_est3:
                    st.metric("% Reservorios", f"{(reservorios/len(resultados)*100):.1f}%")

st.divider()
st.caption("Predictor de Reservorio v2.0 | Fase 2 Midterm - Mejorado con Dataset")

