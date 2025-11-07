import streamlit as st
import torch
from model.model import DigitalTwinModel
import os

st.set_page_config(page_title="Digital Twin Demo", layout="centered")

@st.cache_resource
def load_model():
    model_path = os.path.join("model", "digital_twin_ts.pt")
    # Load TorchScript (CPU)
    return torch.jit.load(model_path, map_location="cpu")

model = load_model()
model.eval()

st.title("🤖 Digital Twin Demo")
st.write("Input sensor values and get predicted health score.")

s1 = st.number_input("Sensor 1", value=10.0)
s2 = st.number_input("Sensor 2", value=20.0)
s3 = st.number_input("Sensor 3", value=30.0)

if st.button("Predict"):
    x = torch.tensor([[s1, s2, s3]], dtype=torch.float32)
    with torch.no_grad():
        y = model(x).item()
    st.metric("Predicted health", f"{y:.4f}")
