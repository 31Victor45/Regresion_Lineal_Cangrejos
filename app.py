# app.py (MODIFICADO para usar el modelo simplificado con límites fijos 5g y 85g y validación)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN DE RUTAS (DEBEN COINCIDIR CON LOS NUEVOS NOMBRES EN preprocess_and_save_simplified_model.py) ---
DATA_FOLDER = 'data_sets'
MODELS_FOLDER = 'models'

LAMBDAS_FILENAME = 'boxcox_lambdas_simplified.joblib' # Cargar el nuevo archivo de lambdas
MODEL_FILENAME = 'linear_regression_model_simplified.joblib' # Cargar el nuevo archivo de modelo
MODEL_METRICS_FILENAME = 'model_metrics_simplified.joblib' # Cargar el nuevo archivo de métricas

# Rutas completas a los archivos que se cargarán
lambdas_path = os.path.join(DATA_FOLDER, LAMBDAS_FILENAME)
model_path = os.path.join(MODELS_FOLDER, MODEL_FILENAME)
metrics_path = os.path.join(MODELS_FOLDER, MODEL_METRICS_FILENAME)

# --- Configuración de Streamlit ---
st.set_page_config(layout="wide", page_title="Análisis de Cangrejos Simplificado")

# --- Carga de recursos (lambdas, modelo, métricas) ---
@st.cache_resource
def load_ml_resources():
    lambdas = None
    models = None
    metrics = {}

    try:
        lambdas = joblib.load(lambdas_path)
        st.sidebar.success("Lambdas de Box-Cox cargados exitosamente (simplificado).")
    except FileNotFoundError:
        st.sidebar.error(f"Error: El archivo de lambdas '{lambdas_path}' no se encontró.")
        st.sidebar.info("Por favor, ejecuta 'preprocess_and_save_simplified_model.py' primero.")
        st.stop()
    except Exception as e:
        st.sidebar.error(f"Error al cargar los lambdas: {e}")
        st.stop()

    try:
        models = joblib.load(model_path)
        st.sidebar.success("Modelos de regresión lineal cargados exitosamente (simplificado).")
    except FileNotFoundError:
        st.sidebar.error(f"Error: El archivo del modelo '{model_path}' no se encontró.")
        st.sidebar.info("Por favor, ejecuta 'preprocess_and_save_simplified_model.py' para entrenar y guardar el modelo.")
        st.stop()
    except Exception as e:
        st.sidebar.error(f"Error al cargar el modelo: {e}")
        st.stop()
    
    try:
        metrics = joblib.load(metrics_path)
        st.sidebar.success("Métricas del modelo cargadas exitosamente (simplificado).")
    except FileNotFoundError:
        st.sidebar.warning(f"Advertencia: El archivo de métricas '{metrics_path}' no se encontró. No se mostrarán las métricas.")
        metrics = {}
    except Exception as e:
        st.sidebar.warning(f"Advertencia: Error al cargar las métricas: {e}")
        metrics = {}

    return lambdas, models, metrics

boxcox_lambdas, trained_models, all_model_metrics = load_ml_resources()

# --- Definición de Features y Targets para el MODELO SIMPLIFICADO ---
MODEL_FEATURES_ORDER = ['new_weight_boxcox'] # Ahora solo una feature
MODEL_TARGETS_BOXCOX_NAMES = [
    'Shucked Weight_boxcox', 'Viscera Weight_boxcox', 'Shell Weight_boxcox'
]
ORIGINAL_TARGET_NAMES = [
    'Shucked Weight', 'Viscera Weight', 'Shell Weight'
]


# --- Función para preparar la entrada del usuario (AHORA SOLO new_weight) ---
def prepare_user_input_for_model(input_weight, lambdas_dict, expected_features_order):
    # La única columna que se transformará con Box-Cox es 'new_weight'
    col = 'new_weight'
    transformed_weight = None

    if col in lambdas_dict:
        val = input_weight
        # Aplicar el mismo offset usado en el preprocesamiento si el valor es <= 0
        # Este offset solo es relevante si el valor de entrada original (antes del boxcox)
        # puede ser <= 0 y se le aplicó un offset en el preprocesamiento.
        # Dado que estamos validando el min_value a 5.0, es menos probable que sea necesario,
        # pero se mantiene para consistencia si el dataset original tiene 0s o negativos.
        if val <= 0:
            offset_val = 1e-6 # Debe ser consistente con preprocess_and_save.py
            val += offset_val
        
        if lambdas_dict[col] == 1.0:
            transformed_weight = val
        else:
            transformed_weight = boxcox([val], lmbda=lambdas_dict[col])[0]
    else:
        st.error(f"Lambda para '{col}' no encontrado. No se puede transformar la entrada.")
        return pd.DataFrame(columns=expected_features_order) # Retorna DataFrame vacío

    # Crear un DataFrame con la feature transformada, asegurando el orden esperado
    final_input_df = pd.DataFrame(columns=expected_features_order)
    final_input_df['new_weight_boxcox'] = [transformed_weight]
    
    return final_input_df

# --- Funciones de Ploteo (sin cambios) ---
def plot_bar_chart(values, labels, title=''):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=labels, y=values, palette="viridis", ax=ax)
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    ax.set_title(title, fontsize=20)
    ax.set_ylabel('Peso (gramos)')
    for i, v in enumerate(values):
        ax.text(i, v + (max(values) * 0.05 if values else 0.01), f"{round(v, 2)}", ha='center', va='bottom', fontsize=12)
    return fig

def generate_pie_chart(values, labels, title=''):
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ['#87CEFA', '#C82A54', '#00FA9A']
    if sum(values) > 0:
        wedges, texts, autotexts = ax.pie(values, colors=colors[:len(values)], autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
        ax.legend(wedges, labels, title="Categorías", loc="center left", bbox_to_anchor=(1.05, 0, 0.3, 1))
    else:
        ax.text(0.5, 0.5, "No hay datos para mostrar", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, color='gray')
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=20)
    plt.setp(autotexts, size=12, weight="bold")
    return fig

# --- Función principal de la aplicación Streamlit ---
def main():
    st.title("Análisis de los Cangrejos 🦀 (Modelo Simplificado por Peso)")

    # Sidebar para la entrada del usuario
    with st.sidebar:
        st.header("Inserta el Peso del Cangrejo")
        
        # Define rangos mínimos y máximos fijos para el input del new_weight
        MIN_NEW_WEIGHT_APP = 5.0 
        MAX_NEW_WEIGHT_APP = 85.0 
        
        # ÚNICA entrada numérica para el peso del cangrejo
        new_weight_input = st.number_input(
            'New Weight (grams)', 
            value=20.0, # Valor inicial sugerido, dentro del rango [5, 85]
            min_value=MIN_NEW_WEIGHT_APP, 
            max_value=MAX_NEW_WEIGHT_APP, 
            step=0.1,
            help=f"Ingresa el peso total del cangrejo en gramos (entre {MIN_NEW_WEIGHT_APP}g y {MAX_NEW_WEIGHT_APP}g, ambos incluidos)."
        )

        # Botón para activar la predicción
        if st.button("Calcular Predicciones"):
            st.session_state.run_prediction = True
            st.session_state.new_weight_value = new_weight_input # Guardar el valor para usarlo en la validación
        else:
            if 'run_prediction' not in st.session_state:
                st.session_state.run_prediction = False
            if 'new_weight_value' not in st.session_state:
                st.session_state.new_weight_value = new_weight_input # Asegurarse de que el valor inicial esté presente


        # Imagen lateral
        current_dir = os.path.dirname(__file__)
        img_path_relative = os.path.join(current_dir, 'img', 'crab1.png') 
        
        try:
            st.image(img_path_relative, caption="", use_container_width=True)
        except FileNotFoundError:
            st.warning("La imagen 'crab1.png' no se encontró en la ruta esperada ('img/').")
            st.image("https://placehold.co/400x300/cccccc/000000?text=Placeholder", caption="Placeholder de imagen", use_container_width=True)


    # Contenido principal de la aplicación
    # Solo ejecuta la lógica de predicción si el botón fue presionado Y el valor está dentro del rango
    if st.session_state.run_prediction:
        # Validar el valor de entrada antes de proceder
        if not (MIN_NEW_WEIGHT_APP <= st.session_state.new_weight_value <= MAX_NEW_WEIGHT_APP):
            st.error(f"Error: El peso ingresado ({st.session_state.new_weight_value}g) está fuera del rango permitido. "
                     f"Por favor, ingresa un valor entre {MIN_NEW_WEIGHT_APP}g y {MAX_NEW_WEIGHT_APP}g (ambos incluidos).")
            st.session_state.run_prediction = False # Resetear para evitar predicciones con valores inválidos
            return # Detener la ejecución para no mostrar resultados incorrectos
            
        # Si la validación es exitosa, procede con la predicción
        new_weight_input_for_prediction = st.session_state.new_weight_value

        # Preparar el input del usuario (SOLO new_weight)
        prepared_input_df = prepare_user_input_for_model(new_weight_input_for_prediction, boxcox_lambdas, MODEL_FEATURES_ORDER)
        
        if not prepared_input_df.empty:
            # Realizar predicciones con cada modelo entrenado
            predicted_transformed_values = {}
            for target_col_boxcox_name in MODEL_TARGETS_BOXCOX_NAMES:
                if target_col_boxcox_name in trained_models:
                    model = trained_models[target_col_boxcox_name]
                    prediction = model.predict(prepared_input_df)[0]
                    predicted_transformed_values[target_col_boxcox_name] = prediction
                else:
                    st.error(f"Modelo para '{target_col_boxcox_name}' no encontrado. No se puede predecir.")
                    predicted_transformed_values[target_col_boxcox_name] = np.nan

            # Transformar inversamente las predicciones a la escala original
            predicted_original_scale_values = []
            for i, target_col_boxcox_name in enumerate(MODEL_TARGETS_BOXCOX_NAMES):
                original_col_name = ORIGINAL_TARGET_NAMES[i]
                transformed_val = predicted_transformed_values[target_col_boxcox_name]
                
                if not np.isnan(transformed_val) and original_col_name in boxcox_lambdas:
                    lmbda = boxcox_lambdas[original_col_name]
                    detransformed_val = inv_boxcox(transformed_val, lmbda)
                    # Asegurar que los valores no sean negativos (los pesos no pueden ser < 0)
                    predicted_original_scale_values.append(max(0, detransformed_val)) 
                else:
                    predicted_original_scale_values.append(np.nan)
                    if not np.isnan(transformed_val):
                        st.warning(f"No se pudo detransformar '{original_col_name}'. Lambda no encontrado o error.")

            # --- Post-procesamiento para asegurar la coherencia de los pesos ---
            sum_predicted_weights = sum(predicted_original_scale_values)
            
            if sum_predicted_weights > 0:
                adjustment_factor = new_weight_input_for_prediction / sum_predicted_weights
                # Aplica el factor de ajuste
                predicted_original_scale_values = [val * adjustment_factor for val in predicted_original_scale_values]
                # Re-asegura no-negatividad después del ajuste, aunque con un factor positivo no debería hacerlos negativos
                predicted_original_scale_values = [max(0, val) for val in predicted_original_scale_values]
                st.info(f"Las predicciones se han ajustado para que su suma sea {round(new_weight_input_for_prediction, 2)}g (asumiendo que 'New Weight' es el peso total).")
            else:
                st.warning("La suma de las predicciones detransformadas es cero o negativa. No se aplicó ajuste.")
                predicted_original_scale_values = [0.0] * len(ORIGINAL_TARGET_NAMES)


            # --- Mostrar resultados y gráficos ---
            if all(not np.isnan(val) for val in predicted_original_scale_values):
                st.subheader(f"Predicciones para {round(new_weight_input_for_prediction, 2)} gramos de peso total")

                col_chart1, col_chart2 = st.columns([1, 1])

                with col_chart1:
                    fig_bar = plot_bar_chart(predicted_original_scale_values, ORIGINAL_TARGET_NAMES,
                                             'Comparación de Pesos Predichos (Gramos)')
                    st.pyplot(fig_bar)

                with col_chart2:
                    fig_pie = generate_pie_chart(predicted_original_scale_values, ORIGINAL_TARGET_NAMES,
                                                 'Distribución de Categorías Predichas')
                    st.pyplot(fig_pie)
                
                # --- Sección de Resumen de Producción ---
                st.subheader("Resumen de la Producción")
                total_meat_crab = predicted_original_scale_values[0] # Shucked Weight
                utility_kitchen = predicted_original_scale_values[0] + predicted_original_scale_values[2] # Shucked Weight + Shell Weight
                waste = predicted_original_scale_values[1] # Viscera Weight

                summary_data = {
                    "Categoría": ["Carne producida 🥩", "Material de aprovechamiento 🍽️", "Desperdicio 🗑️"],
                    "Peso (gramos)": [
                        round(total_meat_crab, 2),
                        round(utility_kitchen, 2),
                        round(waste, 2)
                    ]
                }
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary.style.format(subset=["Peso (gramos)"], formatter="{:.2f}"), hide_index=True, use_container_width=True)

                # --- Sección de Métricas del Modelo en Tabla ---
                if all_model_metrics:
                    st.subheader("Métricas de Evaluación del Modelo (Entrenamiento y Prueba)")
                    metric_data = {
                        "Métrica": [],
                        "Shucked Weight (Entrenamiento)": [], "Shucked Weight (Prueba)": [],
                        "Viscera Weight (Entrenamiento)": [], "Viscera Weight (Prueba)": [],
                        "Shell Weight (Entrenamiento)": [], "Shell Weight (Prueba)": []
                    }
                    for metric_name in ["Error Absoluto Medio (MAE)", "Coeficiente de Determinación (R²)"]:
                        metric_data["Métrica"].append(metric_name)
                        for original_target_name in ORIGINAL_TARGET_NAMES:
                            metric_data[f"{original_target_name} (Entrenamiento)"].append(
                                all_model_metrics.get(original_target_name, {}).get("entrenamiento", {}).get(metric_name, np.nan)
                            )
                            metric_data[f"{original_target_name} (Prueba)"].append(
                                all_model_metrics.get(original_target_name, {}).get("prueba", {}).get(metric_name, np.nan)
                            )
                    df_metrics = pd.DataFrame(metric_data)
                    numeric_cols = [col for col in df_metrics.columns if col != "Métrica"]
                    st.dataframe(df_metrics.style.format(subset=numeric_cols, formatter="{:.2f}"), hide_index=True, use_container_width=True)
                else:
                    st.info("Métricas de evaluación no disponibles (el archivo de métricas no se encontró o hubo un error al cargarlo).")

            else:
                st.error("No se pudieron generar las predicciones o hubo un error en la detransformación. Revisa la consola para más detalles.")
        else:
            st.error("Error al preparar la entrada del usuario para el modelo. Por favor, verifica el valor ingresado.")

    else:
        st.info("Ingresa el peso total del cangrejo y haz clic en 'Calcular Predicciones' para empezar.")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()