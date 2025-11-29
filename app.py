import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# =========================================
# 1. Cargar modelo
# =========================================
@st.cache_resource
def load_model():
    model = joblib.load("best_model_v1.joblib")
    return model

model = load_model()

# =========================================
# 2. Función robusta para leer archivos
# =========================================
def load_input_file(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.name.lower()

    # Excel
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    # CSV: probamos varias combinaciones
    for enc in ["utf-8", "latin-1"]:
        for sep in [",", ";"]:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc, sep=sep)
            except Exception:
                continue

    st.error(
        "❌ No se pudo leer el archivo.\n\n"
        "Prueba guardarlo nuevamente como CSV UTF-8 o como Excel (.xlsx) e intenta de nuevo."
    )
    st.stop()


# =========================================
# 3. Configuración general de la app
# =========================================
st.set_page_config(page_title="Ejecución del Modelo PI", layout="wide")

st.title("🤖 Ejecución del Modelo PI")
st.write(
    "Sube un archivo CSV/XLSX y el modelo preentrenado generará las predicciones correspondientes."
)

st.markdown(
    "> ⚠️ **Importante**: el archivo debe contener las mismas columnas usadas "
    "para entrenar el modelo."
)

# =========================================
# 4. Cargar archivo de usuario
# =========================================
file = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx"])

if file is not None:
    df = load_input_file(file)

    st.subheader("📄 Vista previa del archivo")
    st.dataframe(df.head())
    st.write(f"Filas: **{df.shape[0]}**, Columnas: **{df.shape[1]}**")

    # =========================================
    # 5. Ejecutar modelo
    # =========================================
    if st.button("Ejecutar modelo"):

        # Validar columnas esperadas si el modelo las guarda
        feature_cols = getattr(model, "feature_names_in_", None)

        if feature_cols is not None:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                st.error(
                    "❌ El archivo no contiene todas las columnas necesarias:\n\n"
                    + "\n".join(f"- {m}" for m in missing)
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

        # DataFrame de resultados
        result = df.copy()
        result["prediccion"] = preds

        if probas is not None:
            result["probabilidad_clase_1"] = probas

        # =========================================
        # 🔥 AGREGAR COLUMNA DE ETIQUETA (0=Normal, 1=Fraude)
        # =========================================
        result["etiqueta_prediccion"] = result["prediccion"].map({
            0: "Normal",
            1: "Fraude"
        })

        st.success("✅ Modelo ejecutado correctamente.")

        # =========================================
        # 6. Resultados de la predicción
        # =========================================
        st.subheader("📊 Resultados de la predicción")
        st.dataframe(result.head())

        # =========================================
        # 7. Resumen de los resultados
        # =========================================
        st.subheader("📌 Resumen de los resultados")

        total = len(result)
        n_fraude = (result["prediccion"] == 1).sum()
        n_normal = (result["prediccion"] == 0).sum()

        pct_fraude = (n_fraude / total * 100) if total > 0 else 0
        pct_normal = (n_normal / total * 100) if total > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total registros procesados", total)
        col2.metric("Casos en los que se detectó fraude", n_fraude, f"{pct_fraude:.1f}%")
        col3.metric("Casos que presentan normalidad", n_normal, f"{pct_normal:.1f}%")

        # =========================================
        # 8. Distribución de clases (gráfico)
        # =========================================
        st.subheader("📉 Distribución de clases")

        dist = (
            result["prediccion"]
            .replace({0: "Presentan normalidad", 1: "Se detectó fraude"})
            .value_counts()
        )
        st.bar_chart(dist)

        # =========================================
        # 9. Análisis por segmento (si existe)
        # =========================================
        if "segmento" in result.columns:
            st.subheader("🏷️ Distribución por segmento — Clase 1 (fraude) (%)")
            seg_stats = (
                result.groupby("segmento")["prediccion"]
                .mean()
                .sort_values(ascending=False) * 100
            )
            st.bar_chart(seg_stats)

        # =========================================
        # 10. Top 20 de los casos con mayor probabilidad de fraude
        # =========================================
        if "probabilidad_clase_1" in result.columns:
            st.subheader("🔥 Top 20 de los casos con una predicción más acertada")
            top20 = result.sort_values("probabilidad_clase_1", ascending=False).head(20)
            st.dataframe(top20)

        # =========================================
        # 11. Descargar resultados en Excel (.xlsx)
        # =========================================
        st.subheader("📥 Descargar resultados")

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            result.to_excel(writer, index=False, sheet_name="Resultados")

        st.download_button(
            label="⬇️ Descargar archivo en Excel (.xlsx)",
            data=output.getvalue(),
            file_name="resultados_modelo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
