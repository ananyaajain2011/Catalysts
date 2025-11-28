import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
from orangecontrib.image import ImageEmbedding

# For Orange models
from Orange.data.pandas_compat import table_from_frame
from Orange.classification import LogisticRegressionLearner

st.set_page_config(page_title="Plant Disease Identifier (Orange)", layout="centered")
st.title("🌿 Plant Disease Identifier (Using Orange)")

# --- Step 1: Upload dataset ---
st.header("Step 1: Upload Images for Training")

train_healthy = st.file_uploader("Upload Healthy Leaf Images (ZIP)", type=["zip"], key="healthy")
train_diseased = st.file_uploader("Upload Diseased Leaf Images (ZIP)", type=["zip"], key="diseased")

model_file = "orange_model.pkcls"

def prepare_dataset(healthy_path, diseased_path):
    import zipfile
    import pandas as pd
    import shutil

    base_dir = "orange_data"
    os.makedirs(base_dir, exist_ok=True)

    # Extract healthy
    healthy_dir = os.path.join(base_dir, "healthy")
    if os.path.exists(healthy_dir):
        shutil.rmtree(healthy_dir)
    with zipfile.ZipFile(healthy_path, 'r') as zip_ref:
        zip_ref.extractall(healthy_dir)

    # Extract diseased
    diseased_dir = os.path.join(base_dir, "diseased")
    if os.path.exists(diseased_dir):
        shutil.rmtree(diseased_dir)
    with zipfile.ZipFile(diseased_path, 'r') as zip_ref:
        zip_ref.extractall(diseased_dir)

    # Combine paths & labels
    data = []
    for img_name in os.listdir(healthy_dir):
        data.append({"path": os.path.join(healthy_dir,img_name), "label": "healthy"})
    for img_name in os.listdir(diseased_dir):
        data.append({"path": os.path.join(diseased_dir,img_name), "label": "diseased"})

    df = pd.DataFrame(data)
    return df

# --- Step 2: Train model ---
if st.button("Train Model") and train_healthy and train_diseased:
    st.info("Preparing dataset...")
    df = prepare_dataset(train_healthy, train_diseased)

    st.info("Embedding images...")
    embedder = ImageEmbedding(pretrained_model="VGG16")
    features = embedder(df["path"].tolist())
    df_features = np.array(features)
    
    import pandas as pd
    df_feat = pd.DataFrame(df_features)
    df_feat["label"] = df["label"]

    st.info("Training Orange Logistic Regression model...")
    table = table_from_frame(df_feat)
    X = table.X
    y = table.Y

    learner = LogisticRegressionLearner()
    model = learner(table)

    # Save model
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    st.success("Model trained and saved as orange_model.pkcls!")

# --- Step 3: Upload single image for prediction ---
st.header("Step 3: Predict Leaf Health")
uploaded = st.file_uploader("Upload Leaf Image to Predict", type=["jpg","jpeg","png"])

if uploaded and os.path.exists(model_file):
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Leaf", use_column_width=True)

    embedder = ImageEmbedding(pretrained_model="VGG16")
    features = embedder([uploaded])
    df_feat = pd.DataFrame(features)

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    from Orange.data import Table
    table = Table.from_numpy(domain=model.domain, X=df_feat.values)
    pred = model(table)
    st.write(f"Predicted class: {pred[0].value}")
else:
    if not os.path.exists(model_file):
        st.info("Train the model first to predict.")
    else:
        st.info("Upload an image to predict.")

