# 📑 MATRIKS TINDAK LANJUT REVISI UJIAN HASIL & PANDUAN ASISTENSI

Dokumen ini disusun untuk menjawab seluruh poin revisi penguji pada Ujian Hasil Skripsi/Tugas Akhir:

---

## 📌 TABEL MATRIKS REVISI PENGUJI

| No | Catatan Revisi Dosen Penguji | Akar Masalah Sebelum Revisi | Tindakan Perbaikan yang Telah Dilakukan | Lokasi File Terkait |
|---|---|---|---|---|
| **1** | **Implementasi metode diperhatikan lagi, belum tuntas** | 1. Terdapat formula pseudo-matematis yang keliru di aplikasi (menghubungkan shortcut connection ResNet $F(x) + x$ dengan perhitungan kalori makronutrisi).<br>2. Training pipeline belum mencatat grafik konvergensi (*learning curves*) loss & akurasi per epoch, serta belum menghitung metrik F1-score/precision/recall secara menyeluruh. | 1. **Meluruskan metodologi secara ilmiah:** Deep Learning (ResNet-50 & EfficientNet-B0) murni sebagai *Feature Extractor* & *Classifier* via Softmax. Estimasi kandungan energi dihitung menggunakan **Atwater General Factor System** standar FAO/Kemenkes.<br>2. Menambahkan learning rate scheduler, perekaman riwayat loss/akurasi, dan grafik kurva pembelajaran (*loss & accuracy curves*). | [`model.py`](file:///Users/rizkihiibrahim/Downloads/testing%20dea/model.py)<br>[`train.py`](file:///Users/rizkihiibrahim/Downloads/testing%20dea/train.py)<br>[`app.py`](file:///Users/rizkihiibrahim/Downloads/testing%20dea/app.py) |
| **2** | **Data dan kriteria pengenalan gambar dilengkapi; diarahkan ke kandungan** | Data sebelumnya hanya memuat angka kalori tunggal flat dan teks singkat tanpa indikator biokimia gizi (karbohidrat, protein, lemak, serat, natrium, dsb). | 1. Melengkapi dataset rekomendasi dengan **kandungan gizi lengkap** per porsi referensi (Kalori, Karbohidrat, Protein, Lemak, Serat, Sodium).<br>2. Menetapkan **Kriteria Gizi Terarah** (misal: *Tinggi Protein & Rendah Lemak*, *Tinggi Kalori & Lemak Olahan*).<br>3. Menambahkan visualisasi Donut Chart proporsi makronutrisi & fitur komparasi berbasis target gizi. | [`food_recommendations.csv`](file:///Users/rizkihiibrahim/Downloads/testing%20dea/food_recommendations.csv)<br>[`app.py`](file:///Users/rizkihiibrahim/Downloads/testing%20dea/app.py) |
| **3** | **Lakukan pengujian dengan menggunakan Google Colab, Cek catatan yang saya berikan saat ujian** | Belum tersedianya notebook Jupyter/Colab yang terstruktur untuk pengujian eksperimental dan reproduksibilitas hasil pengujian dosen. | Dibuatkan notebook Google Colab lengkap: **`Pengujian_Model_Food_Classification.ipynb`** yang mencakup GPU training, visualisasi komparasi ResNet-50 vs EfficientNet-B0, Confusion Matrix, Classification Report, dan inferensi gizi Atwater. | [`Pengujian_Model_Food_Classification.ipynb`](file:///Users/rizkihiibrahim/Downloads/testing%20dea/Pengujian_Model_Food_Classification.ipynb) |
| **4** | **Rapikan laporan sesuai kaidah penulisan** | Penjelasan metodologi sistem dan pembahasan metrik evaluasi masih belum sistematis di draft laporan. | Disiapkan draf narasi ilmiah standar akademik untuk Bab 3 (Metodologi) dan Bab 4 (Hasil dan Pembahasan) di bawah ini. | Draf Bab 3 & Bab 4 di bawah ini. |

---

## 🏛️ DRAF MATERI LAPORAN: BAB 3 (METODOLOGI PENELITIAN)

### 3.1 Alur Sistem (*System Flowchart*)
Sistem dirancang dengan arsitektur dua tahap (*Two-Stage Pipeline*):
1. **Tahap 1: Pengenalan Citra (*Computer Vision & Deep Learning*)**
   - Citra makanan masukan di-preprocessing (Resize $224 \times 224$, normalisasi mean $[0.485, 0.456, 0.406]$ dan standard deviation $[0.229, 0.224, 0.225]$).
   - Ekstraksi fitur spasial menggunakan CNN (*Convolutional Neural Network*) dengan arsitektur **ResNet-50** dan **EfficientNet-B0** berbasis Transfer Learning ImageNet.
   - Klasifikasi probabilitas kelas citra menggunakan fungsi aktivasi **Softmax**:
     $$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$
     di mana $z_i$ adalah logit keluaran lapisan *Fully Connected* untuk kelas $i$, dan $C=4$ adalah jumlah kelas pangan.

2. **Tahap 2: Pemodelan Kandungan Gizi & Estimasi Energi (*Nutritional Modeling*)**
   - Menghubungkan kelas makanan terprediksi dengan basis data komposisi gizi terstandarisasi (Tabel Komposisi Pangan Indonesia / USDA).
   - Menghitung nilai energi total metabolis menggunakan **Sistem Faktor Umum Atwater** (*Atwater General Factor System*):
     $$\text{Energi Total (kkal)} = \left[ (4 \times \text{Karbohidrat (g)}) + (4 \times \text{Protein (g)}) + (9 \times \text{Lemak (g)}) \right] \times \frac{\text{Porsi}(\%)}{100}$$
   - Menganalisis kriteria kecukupan gizi (rasio makronutrisi dan rekomendasi diet khusus).

---

## 📊 DRAF MATERI LAPORAN: BAB 4 (HASIL DAN PEMBAHASAN)

### 4.1 Evaluasi Kinerja Arsitektur CNN
Pengujian dilakukan pada dataset citra makanan dengan pembagian data latih (*training*) dan data validasi (*validation*):
- **ResNet-50**: Mengandalkan *Residual Block* dengan *skip connection* $y = \mathcal{F}(x, \{W_i\}) + x$ yang mencegah masalah *vanishing gradient* pada jaringan berkedalaman 50 lapisan.
- **EfficientNet-B0**: Mengandalkan *Compound Scaling* yang menyeimbangkan kedalaman jaringan ($d = \alpha^\phi$), lebar kanal ($w = \beta^\phi$), dan resolusi citra ($r = \gamma^\phi$) dengan blok *MBConv (Inverted Residual Block)*.

### 4.2 Metrik Evaluasi Pengujian
Metrik kuantitatif yang diukur mencakup:
1. **Akurasi Total (*Accuracy*)**: Persentase ketepatan prediksi model terhadap seluruh sampel data uji.
2. **Precision**: Tingkat ketepatan antara citra yang diprediksi dengan data aktual pada kelas tersebut.
3. **Recall (Sensitivitas)**: Kemampuan model menemukan kembali seluruh sampel citra yang benar-benar merupakan kelas tersebut.
4. **F1-Score**: Rata-rata harmonik antara *Precision* dan *Recall* yang menunjukkan keseimbangan kinerja model.

---

## 🗣️ PANDUAN VERBAL SAAT ASISTENSI DENGAN DOSEN

Ketika dosen meminta penjelasan mengenai revisi:

1. **Tentang "Implementasi Metode":**
   > *"Izin Bapak/Ibu, pada implementasi metode sebelumnya terdapat kesalahan pemahaman dalam integrasi rumus. Pada revisi ini, saya telah memisahkan peran metode secara tegas dan ilmiah:*
   > - *ResNet-50 / EfficientNet-B0 murni bertugas mengekstrak fitur spasial citra dan menghasilkan klasifikasi kelas makanan melalui probabilitas Softmax.*
   > - *Setelah kelas makanan teridentifikasi, perhitungan nilai kalori dan distribusinya dihitung menggunakan **Sistem Faktor Atwater (Atwater General Factor System)** terstandarisasi gizi (4 kkal/g karbohidrat, 4 kkal/g protein, dan 9 kkal/g lemak) yang disesuaikan secara proporsional dengan persentase porsi.*
   > - *Proses training sekarang telah dilengkapi pencatatan riwayat kurva Loss & Akurasi (Learning Curves) serta evaluasi Precision, Recall, dan F1-Score."*

2. **Tentang "Data dan Kriteria Kandungan Gizi":**
   > *"Izin Bapak/Ibu, data pengenalan makanan telah kami lengkapi bukan hanya sekadar angka kalori tunggal, melainkan **profil kandungan gizi komprehensif**, meliputi: Karbohidrat, Protein, Lemak, Serat, dan Natrium per porsi referensi. Sistem juga kini dilengkapi kriteria gizi terarah (contoh: Tinggi Protein & Rendah Lemak Jenuh pada Seafood, atau Tinggi Kalori & Lemak Olahan pada Nasi Goreng) beserta visualisasi donut chart makronutrisi dan catatan diet kesehatan."*

3. **Tentang "Pengujian di Google Colab":**
   > *"Untuk pengujian eksperimental sesuai arahan Bapak/Ibu, saya telah menyusun notebook terstruktur **`Pengujian_Model_Food_Classification.ipynb`** yang siap dijalankan di Google Colab menggunakan akselerasi GPU. Di dalam notebook tersebut terdapat pengujian komparasi langsung antara ResNet-50 dan EfficientNet-B0, visualisasi kurva konvergensi pelatihan, matriks kebingungan (Confusion Matrix), serta laporan klasifikasi per kelas."*
