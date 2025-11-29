import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache_resource
def load_model():
    """
    Load the trained model from disk. The model file must be present in the
    same directory as this script.
    """
    return joblib.load("best_model_v1.joblib")


def preprocess_input(inputs: dict) -> pd.DataFrame:
    """
    Preprocess the raw input dictionary into a pandas DataFrame that matches
    the feature set expected by the trained model. This function needs to
    replicate the same feature engineering and preprocessing steps that were
    applied in the training notebook.

    Parameters
    ----------
    inputs : dict
        Raw user inputs keyed by feature name.

    Returns
    -------
    pd.DataFrame
        A DataFrame with a single row, containing the processed features.

    Notes
    -----
    This example includes only placeholder transformations. You should
    implement the actual preprocessing logic based on your notebook. For
    example, performing one-hot encoding, calculating ranges, or mapping
    categorical values exactly as done during training.
    """
    # Construct DataFrame from inputs
    df = pd.DataFrame([inputs])

    # TODO: Replace with real preprocessing steps
    # For now, just return the DataFrame as is
    return df


def main():
    st.set_page_config(page_title="Predicción Modelo PI", layout="wide")
    st.title("📊 Predicción con el Modelo PI")
    st.write(
        "Este aplicativo carga un modelo preentrenado para predecir el posible "
        "evento de caída (clase 1) o no (clase 0)."
    )

    model = load_model()

    st.sidebar.header("Parametros de entrada")

    # Definición de entradas de ejemplo. Ajusta estas variables acorde a
    # las características utilizadas en tu modelo.
    consumo_12m = st.sidebar.number_input(
        "Consumo promedio últimos 12 meses", min_value=0.0, step=1.0, value=100.0
    )
    consumo_24m = st.sidebar.number_input(
        "Consumo promedio últimos 24 meses", min_value=0.0, step=1.0, value=120.0
    )
    consumo_36m = st.sidebar.number_input(
        "Consumo promedio últimos 36 meses", min_value=0.0, step=1.0, value=130.0
    )
    segmento = st.sidebar.selectbox(
        "Segmento", ["Residencial", "Pyme", "Corporativo"], index=0
    )

    if st.sidebar.button("Predecir"):
        # Construye diccionario con los datos introducidos
        raw_data = {
            "consumo_12m": consumo_12m,
            "consumo_24m": consumo_24m,
            "consumo_36m": consumo_36m,
            "segmento": segmento,
            # Añade aquí cualquier otra variable necesaria
        }

        # Preprocesamiento
        X = preprocess_input(raw_data)

        # Predicción
        pred = model.predict(X)[0]
        proba = (
            model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
        )

        st.subheader("Resultado de la predicción")
        if pred == 1:
            st.success("✅ El modelo predice: Caída / Evento positivo (clase 1)")
        else:
            st.info("ℹ️ El modelo predice: Sin caída / Negativo (clase 0)")

        if proba is not None:
            st.write(f"Probabilidad de clase 1: **{proba:.2%}**")

        # Mostrar los datos procesados para depuración
        st.write("\n**Datos procesados:**")
        st.dataframe(X)


if __name__ == "__main__":
    main()
