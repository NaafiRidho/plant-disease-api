import os
import json
import numpy as np
from PIL import Image
import io

# TensorFlow import dengan suppress warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow tidak tersedia. Gunakan mode mock.")

# Path ke model dan label
# backend/utils/ → backend/ → model/
BACKEND_UTILS_DIR = os.path.dirname(__file__)          # backend/utils/
BACKEND_DIR = os.path.dirname(BACKEND_UTILS_DIR)       # backend/
PROJECT_DIR = os.path.dirname(BACKEND_DIR)             # SistemPendeteksiTanaman/
MODEL_DIR = os.path.join(BACKEND_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'plant_disease_model.h5')
LABELS_PATH = os.path.join(MODEL_DIR, 'class_labels.json')

# Global model variable
_model = None
_class_labels = None

# Class labels default (sesuai urutan training ImageDataGenerator alphanumeric)
DEFAULT_CLASS_LABELS = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

IMG_SIZE = (224, 224)


def load_model():
    """Load model TensorFlow dari file .h5"""
    global _model, _class_labels

    # Load class labels
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r') as f:
            _class_labels = json.load(f)
        print(f"[MODEL] Class labels dimuat: {len(_class_labels)} kelas")
    else:
        _class_labels = DEFAULT_CLASS_LABELS
        print(f"[MODEL] Menggunakan default class labels: {len(_class_labels)} kelas")

    # Load model
    if not TF_AVAILABLE:
        print("[MODEL] TensorFlow tidak tersedia, mode mock aktif.")
        return False

    if not os.path.exists(MODEL_PATH):
        print(f"[MODEL] File model tidak ditemukan di: {MODEL_PATH}")
        print("[MODEL] Mode mock aktif. Jalankan train.py terlebih dahulu.")
        return False

    try:
        _model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[MODEL] Model berhasil dimuat dari: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[MODEL] Gagal memuat model: {e}")
        return False


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocessing gambar sebelum prediksi.
    - Resize ke 224x224
    - Convert ke RGB
    - Normalisasi pixel ke [0, 1]
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # Pastikan mode RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize gambar
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        
        # Convert ke numpy array dan normalisasi
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        
        # Tambah dimensi batch: (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Gagal memproses gambar: {str(e)}")


def predict_image(image_bytes: bytes) -> dict:
    """
    Prediksi penyakit tanaman dari bytes gambar.
    
    Returns:
        dict dengan keys:
            - predicted_class: nama kelas prediksi terbaik
            - confidence: confidence score (0-1)
            - top_3: list top 3 prediksi [{class, confidence}]
            - is_mock: True jika menggunakan mode mock
    """
    global _model, _class_labels

    # Inisialisasi label jika belum
    if _class_labels is None:
        load_model()

    # Preprocessing gambar
    img_array = preprocess_image(image_bytes)

    # Mode mock jika model belum tersedia
    if _model is None or not TF_AVAILABLE:
        return _mock_prediction()

    # Prediksi menggunakan model
    try:
        predictions = _model.predict(img_array, verbose=0)
        predictions = predictions[0]  # Ambil batch pertama
        
        # Top-3 prediksi
        top_indices = np.argsort(predictions)[::-1][:3]
        
        top_3 = []
        for idx in top_indices:
            top_3.append({
                "class": _class_labels[idx],
                "confidence": float(predictions[idx]),
                "confidence_percent": round(float(predictions[idx]) * 100, 2)
            })
        
        return {
            "predicted_class": top_3[0]["class"],
            "confidence": top_3[0]["confidence"],
            "confidence_percent": top_3[0]["confidence_percent"],
            "top_3": top_3,
            "is_mock": False
        }
    except Exception as e:
        raise RuntimeError(f"Gagal melakukan prediksi: {str(e)}")


def _mock_prediction() -> dict:
    """Prediksi mock ketika model belum ditraining."""
    import random
    
    # Simulasi prediksi random untuk demo
    random.seed(42)
    mock_classes = random.sample(DEFAULT_CLASS_LABELS, 3)
    confidences = sorted([random.uniform(0.3, 0.95) for _ in range(3)], reverse=True)
    
    # Normalize agar jumlah = 1
    total = sum(confidences)
    confidences = [c / total for c in confidences]
    
    top_3 = [
        {
            "class": mock_classes[i],
            "confidence": confidences[i],
            "confidence_percent": round(confidences[i] * 100, 2)
        }
        for i in range(3)
    ]
    
    return {
        "predicted_class": top_3[0]["class"],
        "confidence": top_3[0]["confidence"],
        "confidence_percent": top_3[0]["confidence_percent"],
        "top_3": top_3,
        "is_mock": True,
        "mock_message": "Model belum ditraining. Hasil ini adalah simulasi. Jalankan train.py terlebih dahulu."
    }


def get_class_labels() -> list:
    """Return daftar semua class labels."""
    global _class_labels
    if _class_labels is None:
        load_model()
    return _class_labels or DEFAULT_CLASS_LABELS


def is_model_loaded() -> bool:
    """Cek apakah model sudah dimuat."""
    return _model is not None
