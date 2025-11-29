import streamlit as st
import pandas as pd
import joblib

# =========================================
# 1. Cargar modelo
# =========================================
@st.cache_resource
def load_model():
    model = joblib.load("best_model_v1.joblib")
    return model

model = load_model()

# =========================================
# 2. Configuración general
# =========================================
st.set_page_config(page_title="Despliegue del Modelo PI", layout="wide")

st.title("🚀 Despliegue del Modelo PI")
st.write(
    "Sube un archivo CSV/XLSX con los datos de interés y el modelo entrenado "
    "aplicará automáticamente sus predicciones."
)

st.markdown(
    "> ⚠️ **Importante**: el archivo debe contener las **mismas columnas** "
    "que se usaron para entrenar el modelo."
)

# =========================================
# 3. Cargar archivo
# =========================================
file = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx"])

if file is not None:

    # Detectar si es CSV o Excel
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("📄 Vista previa del archivo input")
    st.dataframe(df.head())
    st.write(f"Filas: **{df.shape[0]}**, Columnas: **{df.shape[1]}**")

    # =========================================
    # 4. Aplicar modelo
    # =========================================
    if st.button("Ejecutar despliegue del modelo"):
        
        # Si el modelo guarda el listado de features, validamos
        feature_cols = getattr(model, "feature_names_in_", None)

        if feature_cols is not None:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                st.error(
                    "❌ El archivo no contiene todas las columnas necesarias.\n\n"
                    "Faltan estas columnas:\n- " + "\n- ".join(missing)
                )
                st.stop()

            X = df[list(feature_cols)].copy()
        else:
            X = df.copy()

        # Predicciones
        preds = model.predict(X)

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)[:, 1]
        else:
            probas = None

        # Construcción del resultado final
        result = df.copy()
        result["prediccion"] = preds

        if probas is not None:
            result["probabilidad_clase_1"] = probas

        st.success("✅ Despliegue ejecutado y resultados generados.")
        
        st.subheader("📊 Resultados del despliegue")
        st.dataframe(result.head())

        # =========================================
        # 5. Métricas rápidas
        # =========================================
        st.subheader("📌 Resumen del despliegue")

        total = len(result)
        n_pos = (result["prediccion"] == 1).sum()
        n_neg = (result["prediccion"] == 0).sum()
        pct_pos = n_pos / total * 100 if total > 0 else 0
        pct_neg = n_neg / total * 100 if total > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total registros procesados", total)
        col2.metric("Casos clasificados como 1", n_pos, f"{pct_pos:.1f}%")
        col3.metric("Casos clasificados como 0", n_neg, f"{pct_neg:.1f}%")

        # =========================================
        # 6. Distribución de clases (gráfico)
        # =========================================
        st.subheader("📉 Distribución de clases")
        st.bar_chart(result["prediccion"].value_counts())

        # =========================================
        # 7. Análisis por segmento (si existe)
        # =========================================
        if "segmento" in result.columns:
            st.subheader("🏷️ Distribución por segmento — Clase 1 (%)")
            seg_stats = (
                result.groupby("segmento")["prediccion"]
                .mean()
                .sort_values(ascending=False) * 100
            )
            st.bar_chart(seg_stats)

        # =========================================
        # 8. Top casos más probables
        # =========================================
        if "probabilidad_clase_1" in result.columns:
            st.subheader("🔥 Top 20 casos con mayor probabilidad")
            top_20 = result.sort_values("probabilidad_clase_1", ascending=False).head(20)
            st.dataframe(top_20)

        # =========================================
        # 9. Descargar CSV final
        # =========================================
        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar archivo con resultados",
            data=csv_bytes,
            file_name="resultado_despliegue_modelo.csv",
            mime="text/csv",
        )
