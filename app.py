import json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

IMG = 224
MODEL_PATHS = {
    "Transfer Learning (EfficientNetB0)": "/content/drive/MyDrive/Plant_Village_results/models/transfer_efficientnetb0.keras",
    "Custom CNN": "/content/drive/MyDrive/Plant_Village_results/models/cnn.keras",
}

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")


@st.cache_resource
def load_model(path):
    return keras.models.load_model(path, compile=False)


@st.cache_data
def load_classes():
    return json.load(open("results/class_names.json"))


def pretty(name):
    return name.replace("___", " - ").replace("_", " ")


def preprocess(pil_img):
    pil_img = pil_img.convert("RGB").resize((IMG, IMG))
    return np.asarray(pil_img, dtype=np.float32)


def grad_cam(model, img):
    feat = model.get_layer("features")
    head = model.layers[model.layers.index(feat) + 1:]
    pre = model.layers[1:model.layers.index(feat)]
    pre = [l for l in pre if not isinstance(l, keras.Sequential)]   # skip augmentation
    x = tf.convert_to_tensor(img[None])
    for l in pre:
        x = l(x)
    with tf.GradientTape() as tape:
        fmap = feat(x, training=False)
        tape.watch(fmap)
        out = fmap
        for l in head:
            out = l(out, training=False)
        cls = int(tf.argmax(out[0]))
        score = out[:, cls]
    grads = tape.gradient(score, fmap)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * fmap, axis=-1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG, IMG)).numpy().squeeze()
    probs = tf.nn.softmax(out[0]).numpy()
    return cam, probs


def overlay(img, cam, alpha=0.45):
    heat = plt.cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * (img / 255.0) + alpha * heat, 0, 1)


# ---------------- UI ----------------
st.title("Plant Disease Detector")
st.write("Take a photo of a leaf or upload one. The model predicts the disease and Grad-CAM shows what it looked at.")

with st.sidebar:
    model_name = st.selectbox("Model", list(MODEL_PATHS))
    source = st.radio("Image source", ["Upload image", "Camera"])
    show_cam = st.checkbox("Show Grad-CAM", value=True)

model = load_model(MODEL_PATHS[model_name])
class_names = load_classes()

if source == "Upload image":
    file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
else:
    file = st.camera_input("Take a picture of the leaf")

if file is not None:
    pil_img = Image.open(file)
    img = preprocess(pil_img)

    with st.spinner("Analysing..."):
        cam, probs = grad_cam(model, img)

    top = probs.argsort()[::-1][:3]
    best, conf = top[0], probs[top[0]]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(img / 255.0, use_container_width=True)
    with col2:
        if show_cam:
            st.subheader("Grad-CAM")
            st.image(overlay(img, cam), use_container_width=True)

    st.subheader("Prediction")
    if "healthy" in class_names[best].lower():
        st.success(f"{pretty(class_names[best])}  ({conf:.1%})")
    else:
        st.error(f"{pretty(class_names[best])}  ({conf:.1%})")
    if conf < 0.60:
        st.warning("Low confidence. Try a clearer, closer photo of a single leaf on a plain background.")

    st.write("Top 3 predictions")
    for i in top:
        st.progress(float(probs[i]), text=f"{pretty(class_names[i])}: {probs[i]:.1%}")