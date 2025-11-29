# 📘 Modelo PI — Detección de Fraude con Streamlit

Este repositorio contiene todo el flujo necesario para procesar datos, generar un dataset final, entrenar un modelo de Machine Learning y desplegarlo mediante Streamlit para realizar predicciones en tiempo real.

El proyecto está diseñado para ser **100 % reproducible**, permitiendo replicar el procesamiento, el análisis, el entrenamiento del modelo y su uso en producción.

## 📂 Estructura del proyecto

```
.
├── RawData.xlsx                  # Datos originales en crudo
├── salida_extendidafull3.csv    # Dataset procesado final
├── raw-to-dataset.ipynb         # Notebook de procesamiento inicial
├── eda-model-pi.ipynb           # Notebook de EDA + entrenamiento del modelo
├── best_model_v1.joblib         # Modelo entrenado final
├── app.py                       # Aplicación en Streamlit para predicciones
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Documentación del repositorio
```

- **RawData.xlsx**: archivo de datos original en bruto.
- **raw-to-dataset.ipynb**: notebook que transforma los datos crudos en un dataset limpio y listo para análisis.
- **salida_extendidafull3.csv**: dataset resultante del procesamiento.
- **eda-model-pi.ipynb**: notebook donde se explora el dataset, se entrenan modelos y se guarda el mejor.
- **best_model_v1.joblib**: modelo de detección de fraude entrenado y guardado con `joblib`.
- **app.py**: aplicación de Streamlit que permite cargar archivos, generar predicciones y descargar resultados.
- **requirements.txt**: lista de dependencias para reproducir el entorno.

## 🔄 Flujo del proyecto

El pipeline se divide en tres etapas principales:

### 1️⃣ Procesamiento inicial de datos

- **Notebook**: `raw-to-dataset.ipynb`
- **Entrada**: `RawData.xlsx`
- **Salida**: `salida_extendidafull3.csv`

Este notebook realiza:
- Limpieza y estandarización de datos.
- Conversión de tipos de variable.
- Manejo de valores nulos y duplicados.
- Ingeniería de características (si aplica).
- Exportación del dataset procesado a CSV.

### 2️⃣ Análisis exploratorio y entrenamiento del modelo

- **Notebook**: `eda-model-pi.ipynb`
- **Entrada**: `salida_extendidafull3.csv`
- **Salida**: `best_model_v1.joblib`

Incluye:
- Análisis exploratorio de los datos (EDA): distribuciones, correlaciones, outliers y balance de clases.
- Preparación de las features (`feature_names_in_`) y selección de variables.
- Entrenamiento de modelos supervisados.
- Evaluación del desempeño mediante métricas como matriz de confusión, ROC-AUC, F1-score y curva precision–recall.
- Guardado del mejor modelo entrenado en formato Joblib.

### 3️⃣ Aplicación en Streamlit para predicciones

- **Archivo**: `app.py`

Características principales de la aplicación:
- Permite subir archivos CSV o Excel con detección automática de codificación y separadores.
- Valida que el archivo de entrada tenga las columnas requeridas por el modelo.
- Genera predicciones utilizando el modelo entrenado.
- Muestra la probabilidad de pertenecer a la clase de fraude (si el modelo lo permite).
- Agrega una columna de etiqueta legible (`0 = Normal`, `1 = Fraude`) para interpretar fácilmente la predicción.
- Presenta métricas y gráficos de resumen (casos totales, fraudes detectados, distribución por clase, top 20 por probabilidad).
- Permite descargar los resultados en un archivo Excel (`.xlsx`) listo para su uso.

Para ejecutar la aplicación localmente:

```bash
# Instala las dependencias
pip install -r requirements.txt

# Ejecuta la app
streamlit run app.py
```

---

## 📦 Reproducibilidad paso a paso

Siga estos pasos para reproducir el flujo completo:

1. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

2. **Procesar los datos crudos**
   Ejecute el notebook `raw-to-dataset.ipynb` para transformar `RawData.xlsx` en `salida_extendidafull3.csv`.

3. **Entrenar el modelo**
   Ejecute `eda-model-pi.ipynb` para realizar el EDA, entrenar el modelo y guardar `best_model_v1.joblib`.

4. **Iniciar la aplicación**
   ```bash
   streamlit run app.py
   ```

   Al iniciar la app, cargue un archivo que tenga las mismas columnas con las que se entrenó el modelo.

5. **Despliegue en Streamlit Cloud (opcional)**
   Este repositorio está listo para ser desplegado en [Streamlit Cloud](https://share.streamlit.io/):
   - Conecte su cuenta de GitHub.
   - Seleccione este repositorio.
   - Defina `app.py` como archivo principal.
   - La plataforma instalará las dependencias y cargará el modelo automáticamente.

---

## 🧠 Notas y buenas prácticas

- Antes de entrenar un nuevo modelo, actualice y ejecute por completo los notebooks.
- Asegúrese de que el archivo de entrada usado en la app contenga exactamente las columnas con las que se entrenó el modelo; de lo contrario, se producirá un error de validación.
- Mantenga el nombre del modelo coherente (`best_model_vX.joblib`). Si se crea una nueva versión, actualice el nombre y el código de carga en `app.py`.
- Bloquee versiones específicas de librerías en `requirements.txt` para asegurar la reproducibilidad.
- Para reproducir los gráficos de EDA o personalizar la app, modifique directamente los notebooks y `app.py`.

---

