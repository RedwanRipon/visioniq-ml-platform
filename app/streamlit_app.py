import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")   # where the backend runs
DISCLAIMER = ("This demo uses a model trained on the CIFAR-10 benchmark dataset. "
              "Predictions are limited to the supported CIFAR-10 classes and are "
              "intended for demonstration purposes.")

st.set_page_config(page_title="VisionIQ", page_icon="🔎")
st.title("VISIONIQ")
st.subheader("Computer Vision Intelligence Platform")
st.info(DISCLAIMER)

# model information from the API
try:
    info = requests.get(f"{API_URL}/model-info", timeout=5).json()
    st.caption(f"Model: ResNet18 · Version v{info['model_version']} · Dataset: CIFAR-10 · "
               f"Classes: {', '.join(info['classes'])}")
except requests.RequestException:
    st.error(f"Cannot reach the backend at {API_URL}. Is uvicorn running?")
    st.stop()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded is not None:
    st.image(uploaded, caption="Your image", width=250)
    with st.spinner("Classifying..."):
        response = requests.post(f"{API_URL}/predict",
                                 files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                                 timeout=30)

    if response.status_code != 200:
        st.error(f"Error {response.status_code}: {response.json().get('detail')}")
    else:
        result = response.json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", result["predicted_class"].upper())
        c2.metric("Confidence", f"{result['confidence']:.1%}")
        c3.metric("Inference time", f"{result['inference_time_ms']:.0f} ms")

        if result["predicted_class"] == "unknown":
            st.warning("The model is not confident enough (below 90%). "
                       "This image is probably not one of the 10 CIFAR-10 classes.")

        st.markdown("**Top predictions**")
        for i, item in enumerate(result["top_k"], 1):
            st.write(f"{i}. {item['class']} — {item['probability']:.1%}")
            st.progress(item["probability"])

st.divider()
st.caption("Research/demo application. Predictions are limited to the CIFAR-10 classes.")