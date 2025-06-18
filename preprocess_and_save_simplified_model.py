# preprocess_and_save_simplified_model.py (MODIFICADO para filtrar datos de entrenamiento)

import pandas as pd
import numpy as np
from scipy.stats import boxcox
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIGURACIÓN ---
ORIGINAL_CSV_FILENAME = 'crabs.csv'
DATA_FOLDER = 'data_sets'
MODELS_FOLDER = 'models' # Carpeta para guardar los modelos y métricas

# Nombres de los archivos de salida
LAMBDAS_FILENAME = 'boxcox_lambdas_simplified.joblib' # Nuevo nombre para los lambdas
MODEL_FILENAME = 'linear_regression_model_simplified.joblib' # Nuevo nombre para el modelo
MODEL_METRICS_FILENAME = 'model_metrics_simplified.joblib' # Nuevo nombre para las métricas

# Columna numérica del dataset ORIGINAL que será la ÚNICA FEATURE
# y las TARGETS que deben ser transformadas con Box-Cox
COLUMN_TO_TRANSFORM_FEATURE = 'new_weight' # Será la única feature
COLUMNS_TO_TRANSFORM_TARGETS = [
    'Shucked Weight', 'Viscera Weight', 'Shell Weight'
]
# Todas las columnas que necesitan su lambda calculado para la transformación inversa
ALL_COLUMNS_REQUIRING_LAMBDA = [COLUMN_TO_TRANSFORM_FEATURE] + COLUMNS_TO_TRANSFORM_TARGETS

# Nombres de las features y targets Box-Cox transformadas para el modelo
MODEL_FEATURE_BOXCOX = f'{COLUMN_TO_TRANSFORM_FEATURE}_boxcox'
MODEL_TARGETS_BOXCOX_NAMES = [f'{col}_boxcox' for col in COLUMNS_TO_TRANSFORM_TARGETS]

# --- NUEVOS LÍMITES PARA EL ENTRENAMIENTO ---
MIN_WEIGHT_TRAIN = 5.0
MAX_WEIGHT_TRAIN = 85.0

print(f"Iniciando preprocesamiento y entrenamiento para MODELO SIMPLIFICADO...")

# --- CREAR CARPETAS NECESARIAS ---
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# --- Cargar el DataFrame original ---
original_csv_path = os.path.join(DATA_FOLDER, ORIGINAL_CSV_FILENAME)
try:
    df_crabs = pd.read_csv(original_csv_path)
    print(f"DataFrame original '{ORIGINAL_CSV_FILENAME}' cargado exitosamente.")
except FileNotFoundError:
    print(f"ERROR: El archivo '{original_csv_path}' no se encontró.")
    print("Asegúrate de que esté en la carpeta 'data_sets'.")
    # Para desarrollo, puedes crear un DataFrame de prueba si no tienes el archivo
    np.random.seed(42)
    data = {
        'new_weight': np.random.rand(500) * 100 + 1, # Ajustado para simular un rango más amplio inicialmente
        'Shucked Weight': np.random.rand(500) * 20 + 0.1,
        'Viscera Weight': np.random.rand(500) * 10 + 0.1,
        'Shell Weight': np.random.rand(500) * 15 + 0.1,
    }
    df_crabs = pd.DataFrame(data)
    print("DataFrame de prueba creado (usar 'crabs.csv' real en producción).")
    
# Crear una copia del DataFrame para no modificar el original
df_transformed = df_crabs.copy()

# --- FILTRAR EL DATAFRAME POR EL RANGO DE PESO DESEADO ANTES DE CUALQUIER TRANSFORMACIÓN ---
initial_rows = len(df_transformed)
df_transformed = df_transformed[
    (df_transformed[COLUMN_TO_TRANSFORM_FEATURE] >= MIN_WEIGHT_TRAIN) &
    (df_transformed[COLUMN_TO_TRANSFORM_FEATURE] <= MAX_WEIGHT_TRAIN)
].copy() # Usar .copy() para evitar SettingWithCopyWarning
filtered_rows = len(df_transformed)
print(f"Filtrando datos: Se conservaron {filtered_rows} de {initial_rows} filas con '{COLUMN_TO_TRANSFORM_FEATURE}' entre {MIN_WEIGHT_TRAIN}g y {MAX_WEIGHT_TRAIN}g.")

if df_transformed.empty:
    print("ADVERTENCIA: Después de filtrar, el DataFrame está vacío. No se puede entrenar el modelo.")
    print("Asegúrate de que tus datos originales contengan valores dentro del rango de entrenamiento.")
    exit() # Sale del script si no hay datos para entrenar

# Diccionario para almacenar los lambdas de Box-Cox
boxcox_lambdas = {}

print("# Aplicando Transformación Box-Cox y capturando lambdas...")
for var in ALL_COLUMNS_REQUIRING_LAMBDA:
    data_to_transform = df_transformed[var].dropna()
    
    offset = 0.0
    if (data_to_transform <= 0).any():
        print(f"   Advertencia: La columna '{var}' contiene valores no positivos. Ajustando con offset.")
        min_positive = data_to_transform[data_to_transform > 0].min()
        offset = min_positive / 2 if not pd.isna(min_positive) and min_positive > 0 else 1e-6
        data_to_transform = data_to_transform + offset
    
    if len(data_to_transform) > 1 and data_to_transform.nunique() > 1:
        transformed_data, lmbda = boxcox(data_to_transform)
        transformed_series = pd.Series(index=df_transformed.index, dtype=float)
        transformed_series[data_to_transform.index] = transformed_data
        df_transformed[f'{var}_boxcox'] = transformed_series
        boxcox_lambdas[var] = lmbda
        print(f"   Lambda para '{var}': {lmbda:.4f}")
    else:
        df_transformed[f'{var}_boxcox'] = df_transformed[var]
        boxcox_lambdas[var] = 1.0
        print(f"   ADVERTENCIA: Columna '{var}' no apta para Box-Cox. Usando lambda=1.0. Asegúrate de tener suficientes datos variados en el rango filtrado.")

print("Transformación Box-Cox aplicada exitosamente.")

# --- Guardar el diccionario de lambdas ---
lambdas_path_output = os.path.join(DATA_FOLDER, LAMBDAS_FILENAME)
joblib.dump(boxcox_lambdas, lambdas_path_output)
print(f"Diccionario de lambdas guardado en '{lambdas_path_output}' exitosamente.")


# --- Entrenar y Guardar el Modelo ---
print("\n--- Entrenando el modelo de regresión lineal (SIMPLIFICADO) ---")

# Seleccionar la única feature y los targets transformados
# Asegurarse de que X_train y y_train no contengan NaNs si es posible
X = df_transformed[[MODEL_FEATURE_BOXCOX]].dropna()
y = df_transformed.loc[X.index, MODEL_TARGETS_BOXCOX_NAMES].dropna()

# Asegurarse de que los índices coincidan después de dropna
common_index = X.index.intersection(y.index)
X = X.loc[common_index]
y = y.loc[common_index]

if X.empty or y.empty:
    print("ADVERTENCIA: Después de la transformación Box-Cox y manejo de NaNs, el dataset para entrenamiento está vacío.")
    print("No se puede entrenar el modelo. Revisa tus datos y el preprocesamiento.")
    exit()

# Dividir los datos en conjuntos de entrenamiento y prueba (BUENA PRÁCTICA)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

trained_models = {}
all_model_metrics = {}

for target_col_boxcox in MODEL_TARGETS_BOXCOX_NAMES:
    model = LinearRegression()
    # Asegúrate de que las series de y_train y y_test no tengan NaNs antes de entrenar/evaluar
    # Aunque ya hicimos dropna en X e y, es una buena práctica verificar.
    model.fit(X_train, y_train[target_col_boxcox].dropna())
    trained_models[target_col_boxcox] = model

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Asegurarse de que no haya NaNs en las series antes de calcular métricas
    # Esto es importante si hubieran quedado NaNs de alguna manera en y_train/y_test
    y_train_clean = y_train[target_col_boxcox].dropna()
    y_pred_train_clean = pd.Series(y_pred_train, index=X_train.index).loc[y_train_clean.index]

    y_test_clean = y_test[target_col_boxcox].dropna()
    y_pred_test_clean = pd.Series(y_pred_test, index=X_test.index).loc[y_test_clean.index]

    mae_train = mean_absolute_error(y_train_clean, y_pred_train_clean)
    r2_train = r2_score(y_train_clean, y_pred_train_clean)
    mae_test = mean_absolute_error(y_test_clean, y_pred_test_clean)
    r2_test = r2_score(y_test_clean, y_pred_test_clean)
    
    original_target_name = target_col_boxcox.replace('_boxcox', '')
    all_model_metrics[original_target_name] = {
        "entrenamiento": {"Error Absoluto Medio (MAE)": mae_train, "Coeficiente de Determinación (R²)": r2_train},
        "prueba": {"Error Absoluto Medio (MAE)": mae_test, "Coeficiente de Determinación (R²)": r2_test}
    }
    print(f"   Modelo para '{original_target_name}' entrenado. MAE (Prueba): {mae_test:.4f}, R² (Prueba): {r2_test:.4f}")

# Guardar los modelos entrenados
model_path_output = os.path.join(MODELS_FOLDER, MODEL_FILENAME)
joblib.dump(trained_models, model_path_output)
print(f"Modelos de regresión lineal guardados en '{model_path_output}' exitosamente.")

# Guardar las métricas
metrics_path_output = os.path.join(MODELS_FOLDER, MODEL_METRICS_FILENAME)
joblib.dump(all_model_metrics, metrics_path_output)
print(f"Métricas del modelo guardadas en '{metrics_path_output}' exitosamente.")

print("\n--- Proceso completo de preprocesamiento, transformación, entrenamiento y guardado (SIMPLIFICADO) finalizado ---")