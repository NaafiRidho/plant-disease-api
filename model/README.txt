Model placeholder — file model akan digenerate di sini setelah training selesai.

Jalankan training:
  cd ml-model
  pip install -r requirements.txt
  python train.py

Output:
  - plant_disease_model.h5  (model TensorFlow)
  - class_labels.json       (mapping index → nama kelas)
  - training_history.png    (grafik akurasi & loss)
