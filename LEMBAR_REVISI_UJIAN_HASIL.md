# LEMBAR PERTANGGUNGJAWABAN REVISI UJIAN HASIL SKRIPSI

**Judul Penelitian:** Klasifikasi Citra Makanan Menggunakan Deep Learning dan Estimasi Kandungan Gizi Berbasis Sistem Atwater  
**Tanggal Pelaksanaan Ujian:** 18 Agustus 2026 / 16 September 2026  
**Tanggal Selesai Revisi:** 29 September 2026  

---

## 1. RINGKASAN MATRIKS TINDAK LANJUT REVISI

| No | Catatan Revisi Dosen Penguji | Status | File dan Lokasi Perubahan |
|---|---|---|---|
| 1 | Implementasi metode diperhatikan lagi, belum tuntas | Selesai | - `model.py` (Baris 7-38)<br>- `train.py` (Baris 26-160)<br>- `app.py` (Baris 380-425) |
| 2 | Data dan kriteria pengenalan gambar dilengkapi; diarahkan ke kandungan | Selesai | - `food_recommendations.csv` (Baris 1-5)<br>- `app.py` (Baris 45-120, 220-370) |
| 3 | Lakukan pengujian dengan menggunakan Google Colab, Cek catatan yang saya berikan saat ujian | Selesai | - `Pengujian_Model_Food_Classification.ipynb` (27 Sel Langkah Kerja Lengkap) |
| 4 | Rapikan laporan sesuai kaidah penulisan dan asistensikan seluruh masukan | Selesai | - `LEMBAR_REVISI_UJIAN_HASIL.md` (Draf Bab 3 dan Bab 4 Terlampir) |

---

## 2. DETAIL PERUBAHAN BERDASARKAN POIN REVISI

### REVISI 1: IMPLEMENTASI METODE DIPERHATIKAN LAGI, BELUM TUNTAS

#### Masalah Awal:
1. Pada file `app.py` versi sebelumnya, terdapat penulisan formula matematis yang tidak valid secara teori computer vision dan ilmu gizi:
   `F(x, {Wi}) = (karbo * 4) + (protein * 4) + (lemak * 9)` dan `total = F + x_val`.
   Formula tersebut keliru karena mencampuradukkan persamaan fungsi residual konvolusi ResNet `F(x) + x` dengan perhitungan kalori makronutrisi.
2. Pada file `train.py` versi sebelumnya:
   - Dataset tidak menggunakan pemisahan folder latih dan validasi yang telah tersedia (`data/train` dan `data/val`), melainkan melakukan split ulang secara acak.
   - Tidak ada pencatatan riwayat pelatihan per epoch (riwayat loss latih, loss validasi, akurasi latih, akurasi validasi).
   - Metrik evaluasi terbatas hanya pada akurasi tanpa menyajikan Precision, Recall, F1-Score, dan kurva konvergensi (Learning Curves).

#### Tindakan Perbaikan dan Letak File:
1. **File `model.py` (Baris 7-38)**
   - Fungsi `get_model(num_classes, arch, pretrained)` diperbarui agar mendukung transfer learning torchvision modern (`models.ResNet50_Weights.DEFAULT` dan `models.EfficientNet_B0_Weights.DEFAULT`) dengan penanganan fallback yang aman.
   - Penyesuaian lapisan klasifikasi akhir (Linear layer) disesuaikan secara dinamis dengan jumlah kelas target.

2. **File `train.py` (Baris 26-160)**
   - Menambahkan parameter parser `--train_data` dan `--val_data` sehingga alur pelatihan secara eksplisit mengevaluasi data validasi independen (`data/val`).
   - Menerapkan optimizer Adam dengan learning rate scheduler `ReduceLROnPlateau` untuk menurunkan learning rate saat validasi loss stagnan.
   - Menyimpan checkpoint model berkinerja tertinggi ke file `checkpoints/best_model.pth`.
   - Menyimpan seluruh riwayat metrik pelatihan ke file `checkpoints/training_history.json`.
   - Mengenerate grafik kurva pembelajaran resmi `checkpoints/learning_curves.png` (Grafik Loss Latih vs Validasi dan Akurasi Latih vs Validasi).
   - Menyimpan hasil evaluasi matriks kebingungan ke `checkpoints/best_confusion_matrix.png` dan laporan klasifikasi detail ke `checkpoints/best_classification_report.txt`.

3. **File `app.py` (Baris 380-425)**
   - Formula keliru lama telah dihapus sepenuhnya.
   - Metodologi disusun menjadi dua tahapan sistematis (Two-Stage Pipeline):
     - **Tahap 1 (Computer Vision):** Ekstraksi fitur citra melalui CNN (ResNet-50 atau EfficientNet-B0) yang menghasilkan probabilitas kelas dengan fungsi aktivasi Softmax:
       $$P(y = i \mid \mathbf{x}) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$
     - **Tahap 2 (Nutritional Modeling):** Pemetaan kelas citra ke basis data gizi standar, lalu total nilai energi dihitung menggunakan formula fisiologis Sistem Faktor Umum Atwater:
       $$\text{Energi Total (kkal)} = \left[ (4 \times \text{Karbohidrat}) + (4 \times \text{Protein}) + (9 \times \text{Lemak}) \right] \times \frac{\text{Porsi}(\%)}{100}$$

---

### REVISI 2: DATA DAN KRITERIA PENGENALAN GAMBAR DILENGKAPI; DIARAHKAN KE KANDUNGAN

#### Masalah Awal:
Dataset referensi makanan pada sistem sebelumnya hanya memuat angka estimasi kalori tunggal (flat number) tanpa memuat profil kandungan zat gizi biokimia dan tanpa kriteria kesehatan terukur.

#### Tindakan Perbaikan dan Letak File:
1. **File `food_recommendations.csv` (Baris 1-5)**
   Dataset diperluas dengan menambahkan atribut kandungan nutrisi lengkap per porsi acuan terstandarisasi:
   - `class_name`: Nama kelas makanan (Bread, nasi-goreng, Noodles-Pasta, Seafood).
   - `calories_per_portion`: Energi total standar (kkal).
   - `standard_portion_desc`: Deskripsi takaran porsi acuan (contoh: 1 piring sedang 250g, 2 lembar 100g, 1 mangkuk 200g).
   - `carbohydrates_g`: Kandungan karbohidrat dalam satuan gram.
   - `protein_g`: Kandungan protein dalam satuan gram.
   - `fat_g`: Kandungan lemak total dalam satuan gram.
   - `fiber_g`: Kandungan serat pangan dalam satuan gram.
   - `sodium_mg`: Kandungan natrium/garam dalam satuan miligram.
   - `health_score`: Indeks kelayakan gizi (skala 1-100).
   - `nutrition_criteria`: Kriteria gizi terarah (contoh: "Tinggi Protein & Rendah Lemak Jenuh", "Tinggi Kalori & Lemak Olahan", "Padat Energi & Karbohidrat Olahan").
   - `recommendation`: Saran konsumsi ilmiah sesuai pedoman gizi seimbang.
   - `dietary_notes`: Catatan klinis untuk kondisi khusus (diabetes, hipertensi, dislipidemia, hiperurisemia).

2. **File `app.py` (Baris 45-120 dan Baris 220-370)**
   - Antarmuka pengguna diperbarui untuk menampilkan tabel ringkasan kandungan gizi lengkap.
   - Ditambahkan visualisasi Donut Chart komposisi energi makronutrisi (distribusi kalori dari karbohidrat, protein, dan lemak) untuk setiap citra yang dianalisis.
   - Ditambahkan slider penyesuaian porsi (25% hingga 300%) yang secara matematis menghitung ulang seluruh nilai zat gizi secara proporsional.
   - Ditambahkan fitur komparasi citra makanan berbasis kriteria gizi:
     - Gizi Paling Seimbang (Skor Kesehatan Tertinggi)
     - Kandungan Protein Tertinggi
     - Kalori Paling Rendah (Defisit Kalori)
     - Kandungan Lemak Paling Rendah

---

### REVISI 3: PENGUJIAN MENGGUNAKAN GOOGLE COLAB

#### Masalah Awal:
Sebelumnya belum tersedia berkas Google Colab (`.ipynb`) yang terstruktur untuk pengujian eksperimen yang dapat diverifikasi oleh dosen penguji.

#### Tindakan Perbaikan dan Letak File:
Telah dibuat berkas notebook Google Colab resmi:
**`Pengujian_Model_Food_Classification.ipynb`**

Notebook ini terdiri dari 27 sel kode dan teks akademis terstruktur (bebas dari penggunaan simbol emoji) dengan tahapan sebagai berikut:
1. **Langkah 1:** Verifikasi akselerasi perangkat keras GPU (`!nvidia-smi`) dan versi PyTorch.
2. **Langkah 2:** Pemuatan dan inspeksi distribusi dataset (visualisasi diagram batang jumlah sampel data latih dan validasi per kelas).
3. **Langkah 3:** Definisi basis data kandungan gizi dan implementasi Sistem Faktor Atwater.
4. **Langkah 4:** Pipeline pemrosesan data (Resize 224x224, RandomHorizontalFlip, RandomRotation 15 derajat, ColorJitter, normalisasi ImageNet) dan DataLoader PyTorch.
5. **Langkah 5:** Konstruksi arsitektur transfer learning (ResNet-50 dan EfficientNet-B0).
6. **Langkah 6:** Fungsi pelatihan terstandarisasi dengan pencatatan riwayat Loss, Akurasi, dan perbaikan learning rate.
7. **Langkah 7:** Pelatihan Model 1 (ResNet-50) selama 10 epoch dan penyimpanan model terbaik.
8. **Langkah 8:** Pelatihan Model 2 (EfficientNet-B0) selama 10 epoch dengan parameter identik untuk perbandingan yang setara (fair comparison).
9. **Langkah 9:** Plotting grafik komparasi kurva pembelajaran (*Learning Curves: Validation Loss & Validation Accuracy*).
10. **Langkah 10:** Evaluasi Confusion Matrix berdampingan dan Classification Report (Precision, Recall, F1-Score per kelas).
11. **Langkah 11:** Tabel rekapitulasi perbandingan performa ResNet-50 vs EfficientNet-B0.
12. **Langkah 12:** Pengujian inferensi pada sampel citra data uji acak disertai perhitungan estimasi kandungan gizi Atwater.
13. **Langkah 13:** Ekspor dan penyimpanan file bobot model `.pth`, grafik PNG, dan tabel CSV hasil pengujian.

---

### REVISI 4: PERAPIAN LAPORAN SESUAI KAIDAH PENULISAN

Berikut adalah draf materi laporan yang dapat langsung dimasukkan ke dalam naskah skripsi:

#### Draf Bab 3: Metodologi Penelitian
Sistem pengenalan citra makanan dan estimasi kandungan gizi dirancang menggunakan arsitektur dua tahap:
1. **Tahap Ekstraksi Fitur dan Klasifikasi Citra:**
   Model memanfaatkan arsitektur Convolutional Neural Network (CNN) dengan pendekatan Transfer Learning, membandingkan ResNet-50 yang memanfaatkan koneksi lewatan residual (*residual shortcut connection*) dan EfficientNet-B0 yang memanfaatkan penskalaan majemuk (*compound scaling*). Vektor fitur keluaran lapisan konvolusi dihubungkan ke lapisan penentu klasifikasi (*fully connected layer*) yang dipetakan ke fungsi Softmax:
   $$P(y = i \mid \mathbf{x}) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$
   di mana $z_i$ menyatakan nilai logit untuk kelas ke-$i$ dan $C$ menyatakan jumlah kelas makanan ($C=4$).

2. **Tahap Pemodelan dan Estimasi Kandungan Gizi:**
   Berdasarkan kelas makanan hasil klasifikasi berprobabilitas tertinggi, sistem mengaitkan citra dengan basis data gizi referensi terstandarisasi. Nilai energi metabolis total dihitung menggunakan Sistem Faktor Umum Atwater:
   $$\text{Energi Total (kkal)} = \left[ (4 \times \text{Karbohidrat (g)}) + (4 \times \text{Protein (g)}) + (9 \times \text{Lemak (g)}) \right] \times \frac{\text{Porsi}(\%)}{100}$$
   Pemberian bobot 4 kkal/g untuk karbohidrat dan protein, serta 9 kkal/g untuk lemak mengacu pada ketetapan Food and Agriculture Organization (FAO) dan Kementerian Kesehatan Republik Indonesia.

#### Draf Bab 4: Hasil dan Pembahasan
1. **Analisis Kinerja Arsitektur Model:**
   Pengujian model dilakukan menggunakan akselerasi GPU Google Colab pada dataset makanan yang terbagi menjadi data latih dan data validasi independen. Evaluasi dilakukan secara kuantitatif melalui metrik Akurasi, Macro Precision, Macro Recall, dan Macro F1-Score.
2. **Evaluasi Matriks Kebingungan (Confusion Matrix):**
   Confusion matrix menyajikan pemetaan antara kelas aktual dan kelas prediksi, memperlihatkan tingkat kesalahan klasifikasi (*misclassification*) pada kelas dengan kemiripan visual tinggi (seperti Bread dan Noodles-Pasta).
3. **Analisis Estimasi Kandungan Gizi:**
   Integrasi hasil inferensi klasifikasi citra dengan sistem faktor Atwater berhasil menghasilkan estimasi nilai kalori dan makronutrisi yang proporsional terhadap takaran porsi yang diuji, membuktikan bahwa penambahan kriteria gizi memberikan nilai aplikatif yang lebih terarah dibandingkan estimasi kalori tunggal.

---

## 3. PANDUAN PENYAMPAIAN SAAT ASISTENSI DENGAN DOSEN

Saat melakukan asistensi perbaikan revisi dengan dosen pembimbing atau penguji, sampaikan poin-poin berikut secara lugas dan ilmiah:

1. Mengenai implementasi metode:
   "Terima kasih atas masukannya Bapak/Ibu. Pada revisi ini, implementasi metode telah kami selesaikan dan pisahkan secara tegas:
   Arsitektur deep learning ResNet-50 dan EfficientNet-B0 difungsikan secara murni untuk ekstraksi fitur spasial dan klasifikasi citra via Softmax. Sedangkan perhitungan nilai kalori makanan dihitung terpisah menggunakan formula baku Sistem Faktor Umum Atwater berbasis kandungan gramasi makronutrisi."

2. Mengenai kelengkapan data dan kriteria kandungan gizi:
   "Sesuai arahan Bapak/Ibu, basis data pengenalan makanan telah kami lengkapi bukan hanya estimasi kalori, melainkan profil gizi lengkap (Karbohidrat, Protein, Lemak, Serat, dan Natrium) serta kriteria kandungan gizi terarah pada antarmuka sistem."

3. Mengenai pengujian di Google Colab:
   "Seluruh pengujian eksperimental model telah kami susun dan jalankan pada Google Colab melalui notebook Pengujian_Model_Food_Classification.ipynb. Di dalam notebook tersebut telah tersedia perbandingan lengkap antara ResNet-50 dan EfficientNet-B0, kurva konvergensi pelatihan, confusion matrix, dan classification report."
