import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from huggingface_hub import hf_hub_download

# ================= CSS THEME =================
page_theme = """
<style>
/* ===== Background halaman ===== */
[data-testid="stAppViewContainer"] {
    background-color: #E8F5E9;
}

/* Hilangkan background header bawaan */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Sidebar warna senada */
[data-testid="stSidebar"] {
    background-color: rgba(200, 230, 201, 0.95);
}

/* ===== Box utama ===== */
.block-container {
    background-color: #E8F5E9; 
    padding: 2rem;
    border-radius: 15px;
}

/* ===== Teks lebih kontras ===== */
h1, h2, h3, h4, h5, h6, p, label, span {
    color: #1B5E20 !important; 
}

/* ===== Kotak upload file ===== */
[data-testid="stFileUploader"] section {
    background-color: #C8E6C9; 
    border: 2px dashed #2E7D32;
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="stFileUploader"] section:hover {
    background-color: #A5D6A7;
}

/* Tombol browse file */
[data-testid="stFileUploader"] button {
    background-color: #2E7D32;
    color: white;
    border-radius: 8px;
    border: none;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #1B5E20;
}

/* ===== Nama file & ukuran file jadi hitam ===== */
[data-testid="stFileUploaderFileName"] {
    color: black !important;
    font-weight: normal !important;
}
[data-testid="stFileUploaderFileSize"] {
    color: black !important;
}

/* ===== Progress bar ===== */
[data-testid="stProgress"] div > div {
    background-color: #2E7D32 !important;
}

/* ===== Metric box ===== */
[data-testid="stMetricValue"] {
    color: #2E7D32 !important;
    font-weight: bold;
}
[data-testid="stMetricLabel"] {
    color: #1B5E20 !important;
}

/* ===== Pesan sukses ===== */
.stAlert {
    background-color: #C8E6C9;
    color: #1B5E20;
    border-radius: 8px;
}
</style>
"""
st.markdown(page_theme, unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_my_model():
    model_path = hf_hub_download(
        repo_id="coconud/garbage-classification-VGG16-model",
        filename="model_VGG16.keras",
        token=st.secrets["HF_TOKEN"] # Token dari Hugging Face
    )
    return load_model(model_path)

model = load_my_model()

class_names = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

# ================= UI =================
st.title("♻️ Garbage Classification AI")
st.write("Upload gambar untuk mendeteksi jenis sampah dengan model **VGG16**")

uploaded_file = st.file_uploader("📤 Upload Gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(img.resize((224, 224)), caption="📷 Gambar yang diupload", use_container_width=False)

    st.write("### 🔄 Preprocessing...")

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediksi
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    pred_label = class_names[pred_index].upper()
    confidence = float(predictions[0][pred_index])

    # Hasil
    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.metric("Prediksi", pred_label)
    col2.metric("Confidence", f"{confidence:.2%}")

    st.progress(confidence)

    if confidence > 0.8:
        st.success("✅ Model sangat yakin dengan prediksi ini!")
    elif confidence > 0.5:
        st.warning("⚠️ Model cukup yakin, tapi masih ada kemungkinan salah.")
    else:
        st.error("❌ Model kurang yakin, coba gunakan gambar yang lebih jelas.")



