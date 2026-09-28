# 🥗 Food Classification & Nutrition Estimator (Deep Learning)

Sistem Klasifikasi Citra Makanan Berbasis Deep Learning (ResNet-50 vs EfficientNet-B0) dan Estimasi Kandungan Gizi Menggunakan Sistem Faktor Umum Atwater.

---

## 📁 Struktur Proyek
```text
.
├── app.py                                   # Aplikasi antarmuka Streamlit (Inference & UI Kandungan Gizi)
├── model.py                                 # Definisi arsitektur model (ResNet-50 & EfficientNet-B0)
├── train.py                                 # Skrip pelatihan lokal & logging kurva evaluasi
├── utils.py                                 # Dataset loader & utility fungsi
├── food_recommendations.csv                 # Basis data kandungan gizi & kriteria nutrisi lengkap
├── style.css                                # Tema & styling antarmuka Streamlit
├── Pengujian_Model_Food_Classification.ipynb# Notebook pengujian lengkap untuk Google Colab (GPU)
├── MATRIKS_REVISI_DAN_PANDUAN_ASISTENSI.md  # Lembar tindak lanjut revisi penguji & draf skripsi Bab 3 & 4
├── checkpoints/                             # Folder bobot model (.pth) & matriks evaluasi
└── data/                                    # Dataset citra (train & val)
    ├── train/
    └── val/
```

---

## 🚀 1. Menjalankan Pengujian di Google Colab
Sesuai arahan dosen penguji:
1. Buka [Google Colab](https://colab.research.google.com/).
2. Pilih tab **Upload**, lalu unggah file:
   [`Pengujian_Model_Food_Classification.ipynb`](Pengujian_Model_Food_Classification.ipynb).
3. Pastikan runtime menggunakan GPU: **Runtime** -> **Change runtime type** -> **T4 GPU**.
4. Unggah folder/zip dataset `data/` ke Colab atau hubungkan ke Google Drive.
5. Jalankan seluruh sel untuk melatih, membandingkan ResNet-50 vs EfficientNet-B0, serta mencetak kurva evaluasi dan confusion matrix.

---

## 💻 2. Menjalankan Aplikasi Web Streamlit (Lokal)
Jalankan perintah berikut di terminal:
```bash
python -m streamlit run app.py
```
Fitur yang tersedia di aplikasi:
- Pemilihan model (.pth) secara dinamis di sidebar.
- Evaluasi langsung pada data validasi (Confusion Matrix & Classification Report).
- Deteksi citra makanan tunggal atau banyak gambar sekaligus.
- Penyesuaian porsi interaktif (50% - 300%).
- Rincian kandungan gizi lengkap (Kalori, Karbohidrat, Protein, Lemak, Serat, Sodium).
- Visualisasi Donut Chart pembagian kalori makronutrisi.
- Rekomendasi komparatif (Paling Seimbang, Protein Tertinggi, Kalori Terendah).

---

## 🏋️ 3. Melatih Model Secara Lokal (Opsional)
Untuk melatih ulang model dengan logging kurva evaluasi lengkap:

**Menggunakan EfficientNet-B0:**
```bash
python train.py --train_data data/train --val_data data/val --arch efficientnet_b0 --epochs 15
```

**Menggunakan ResNet-50:**
```bash
python train.py --train_data data/train --val_data data/val --arch resnet50 --epochs 15
```
Hasil pelatihan (model terbaik, grafik kurva loss/akurasi, dan confusion matrix) akan otomatis tersimpan di folder `checkpoints/`.