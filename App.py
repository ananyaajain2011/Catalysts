import streamlit as st
from PIL import Image

st.set_page_config(page_title="Plant Disease Identifier", layout="centered")

st.title("🌿 Plant Disease Identifier")
st.write("Upload a leaf image to check plant health. (Demo Version)")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Leaf", use_column_width=True)

    st.subheader("Prediction Result")
    st.success("🌱 Healthy (Demo Prediction)")
    st.caption("Note: This is a demo app. Add a real ML model later.")
else:
    st.info("Please upload an image.")
