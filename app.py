import streamlit as st
import torch
import os

st.set_page_config(page_title="Digital Twin Demo", layout="centered")

@st.cache_resource
def load_model():
    model_path = os.path.join("model", "digital_twin_ts.pt")
    return torch.jit.load(model_path, map_location="cpu")

model = load_model()
model.eval()

st.title("🤖 Digital Twin Simulation")
st.write("Predict system behavior from sensor inputs.")

col1, col2, col3 = st.columns(3)
s1 = col1.number_input("Sensor 1", value=10.0)
s2 = col2.number_input("Sensor 2", value=20.0)
s3 = col3.number_input("Sensor 3", value=30.0)

placeholder = st.empty()

if st.button("Predict"):
    x = torch.tensor([[s1, s2, s3]], dtype=torch.float32)
    with torch.no_grad():
        y = model(x).item()
    placeholder.success(f"Predicted Output: {y:.4f}")
else:
    placeholder.info("Press **Predict** to simulate output.")
