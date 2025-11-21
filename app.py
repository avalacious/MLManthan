import io
import pickle
from pathlib import Path
import wave

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


APP_TITLE = "Anomaly Dashboard (Isolation Forest + OC-SVM)"
DATA_INFO = (
    "Upload a `.dat` file containing only numeric columns. "
    "The raw values will be scaled with the saved `scaler_10batch.pkl` "
    "before running both anomaly detectors."
)


def load_pickle_file(pickle_path: Path):
    try:
        with pickle_path.open("rb") as handle:
            return pickle.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing artifact: {pickle_path.name}") from exc
    except pickle.UnpicklingError as exc:
        raise RuntimeError(
            f"Could not load {pickle_path.name}. The file looks corrupted or was "
            "created with an incompatible Python/NumPy version. "
            "Re-export the artifact and try again."
        ) from exc


@st.cache_resource
def load_artifacts():
    base_dir = Path(__file__).resolve().parent
    iso_model = load_pickle_file(base_dir / "iso_forest_10batch.pkl")
    ocsvm_model = load_pickle_file(base_dir / "ocsvm_model_10batch.pkl")
    scaler = load_pickle_file(base_dir / "scaler_10batch.pkl")
    return iso_model, ocsvm_model, scaler


def read_dat_file(uploaded_file) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            uploaded_file,
            header=None,
            sep=None,
            engine="python",
        )
    except Exception as exc:
        raise ValueError(
            "Unable to parse the .dat file. Make sure it only contains "
            "numeric values separated by spaces, commas, or tabs."
        ) from exc

    df.columns = [f"feature_{idx}" for idx in range(df.shape[1])]
    return df


def synthesize_alert_sound(duration_seconds: float = 0.4, frequency_hz: int = 880):
    sample_rate = 44100
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    tone = 0.5 * np.sin(2 * np.pi * frequency_hz * t)
    audio = np.int16(tone * 32767)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())

    buffer.seek(0)
    return buffer


def make_bar_chart(series: pd.Series, title: str):
    counts = series.value_counts().rename_axis("category").reset_index(name="count")
    fig = px.bar(
        counts,
        x="category",
        y="count",
        text="count",
        title=title,
        color="category",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_title="Count", xaxis_title="", uniformtext_mode="hide")
    return fig


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🚨", layout="wide")
    st.title(APP_TITLE)
    st.caption(DATA_INFO)

    uploaded_file = st.file_uploader("Upload your `.dat` file", type=["dat"])

    if not uploaded_file:
        st.info("Waiting for a `.dat` file...")
        return

    try:
        raw_df = read_dat_file(uploaded_file)
    except ValueError as err:
        st.error(str(err))
        return

    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head())

    iso_model, ocsvm_model, scaler = load_artifacts()
    scaled_values = scaler.transform(raw_df.values)

    iso_pred = iso_model.predict(scaled_values)
    ocsvm_pred = ocsvm_model.predict(scaled_values)

    iso_anomaly = iso_pred == -1
    ocsvm_anomaly = ocsvm_pred == -1

    combined_anomaly = iso_anomaly | ocsvm_anomaly
    common_outlier = iso_anomaly & ocsvm_anomaly
    iso_unique = iso_anomaly & ~ocsvm_anomaly
    ocsvm_unique = ocsvm_anomaly & ~iso_anomaly

    iso_display_value = np.where(iso_anomaly, 1.0, 0.5)

    results_df = pd.DataFrame(
        {
            "IsolationForestValue": iso_display_value,
            "IsolationForestLabel": np.where(iso_anomaly, "Outlier", "Inlier"),
            "OCSVMLabel": np.where(ocsvm_anomaly, "Outlier", "Inlier"),
            "CombinedAnomaly": np.where(combined_anomaly, "Anomaly", "Normal"),
            "CommonOutlier": np.where(common_outlier, "Yes", "No"),
            "IsolationForestOnly": np.where(iso_unique, "Yes", "No"),
            "OCSVMOnly": np.where(ocsvm_unique, "Yes", "No"),
        }
    )

    st.subheader("Model Decisions")
    st.dataframe(results_df)

    if combined_anomaly.any():
        st.error("⚠️ Anomaly detected! Review the highlighted rows.")
        st.toast("Anomaly detected!", icon="⚠️")
        st.audio(
            synthesize_alert_sound(),
            format="audio/wav",
            sample_rate=44100,
        )
    else:
        st.success("No anomalies detected by either model.")

    st.subheader("Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            make_bar_chart(results_df["CombinedAnomaly"], "Total Anomalies (Combined)"),
            use_container_width=True,
        )
        st.plotly_chart(
            make_bar_chart(results_df["IsolationForestLabel"], "Isolation Forest Anomalies"),
            use_container_width=True,
        )
        st.plotly_chart(
            make_bar_chart(results_df["IsolationForestOnly"], "Unique Isolation Forest Outliers"),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            make_bar_chart(results_df["OCSVMLabel"], "OC-SVM Anomalies"),
            use_container_width=True,
        )
        st.plotly_chart(
            make_bar_chart(results_df["CommonOutlier"], "Common Outliers"),
            use_container_width=True,
        )
        st.plotly_chart(
            make_bar_chart(results_df["OCSVMOnly"], "Unique OC-SVM Outliers"),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown(
        "Tip: run the app with `streamlit run app.py`, then upload a `.dat` file "
        "containing the same feature order used during training."
    )


if __name__ == "__main__":
    main()

