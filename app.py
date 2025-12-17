import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="CNN Pneumonia Detection",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "CNN_Classification_1.h5"
IMG_SIZE = 150  # Sesuai model training
CLASS_NAMES = ["Normal", "Pneumonia"]

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_cnn_model():
    model = load_model(MODEL_PATH)
    return model

model = load_cnn_model()

# ===============================
# SIDEBAR INFO
# ===============================
st.sidebar.title("ℹ️ Tentang Aplikasi")
st.sidebar.info(
    """
    Aplikasi ini menggunakan **Convolutional Neural Network (CNN)**
    untuk mendeteksi **Pneumonia pada citra X-Ray dada**.

    ⚠️ **Disclaimer:** Hasil prediksi hanya untuk edukasi dan penelitian,
    **bukan diagnosis medis**.

    **Instruksi:**
    1. Upload citra X-Ray (.jpg / .png)
    2. Tunggu hasil prediksi dan probabilitas muncul.
    """
)

# ===============================
# HEADER
# ===============================
st.title("🫁 CNN Pneumonia Detection")
st.markdown(
    """
    Deteksi **Pneumonia dari X-Ray Dada** secara otomatis menggunakan CNN.
    Upload citra dan sistem akan menampilkan hasil prediksi beserta probabilitasnya.
    """
)
st.divider()

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader(
    "📤 Upload Citra X-Ray (JPG/PNG)",
    type=["jpg", "jpeg", "png",'webp']
)

# ===============================
# PREDICTION
# ===============================
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    # Tampilkan gambar
    with col1:
        image = Image.open(uploaded_file).convert("L")  # GRAYSCALE
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (150,150,1)
    img_array = np.expand_dims(img_array, axis=0)   # (1,150,150,1)

    # Prediksi (binary classifier)
    pred_prob = model.predict(img_array)[0][0]  # output sigmoid (0-1)

    if pred_prob > 0.5:
        class_name = "Pneumonia"
        confidence = pred_prob * 100
        st_color = "red"
    else:
        class_name = "Normal"
        confidence = (1 - pred_prob) * 100
        st_color = "green"

    # Tampilkan hasil prediksi
    with col2:
        st.subheader("🧪 Hasil Prediksi")
        if class_name == "Pneumonia":
            st.markdown(f"<h2 style='color:red'>{class_name}</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:green'>{class_name}</h2>", unsafe_allow_html=True)

        st.metric(label="Confidence", value=f"{confidence:.2f}%")

        st.subheader("📊 Probabilitas Kelas")
        # Normal
        normal_prob = float(1 - pred_prob)
        st.write(f"Normal: {normal_prob*100:.2f}%")
        st.progress(normal_prob)

        # Pneumonia
        pneu_prob = float(pred_prob)
        st.write(f"Pneumonia: {pneu_prob*100:.2f}%")
        st.progress(pneu_prob)


st.divider()

# ===============================
# FOOTER
# ===============================
st.markdown(
    """
    ---
    ⚠️ **Catatan:** Hasil prediksi hanya untuk keperluan edukasi dan penelitian.
    Tidak digunakan sebagai alat diagnosis medis.
    Developed by Your Name.
    """
)
