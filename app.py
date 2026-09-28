import streamlit as st
from PIL import Image
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import csv

from model import get_inference_transforms, load_checkpoint, DEVICE
from utils import make_file_label_list, FoodDataset

# ============================================================
# Page config & styling
# ============================================================
st.set_page_config(
    page_title="Sistem Pengenalan Citra Makanan & Analisis Kandungan Gizi",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS jika tersedia
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
        <h1>Sistem Pengenalan Makanan & Estimasi Kandungan Gizi</h1>
        <p>Klasifikasi Citra Berbasis Deep Learning (ResNet-50 / EfficientNet-B0) & Analisis Komposisi Gizi (Sistem Atwater)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

VAL_DIR = "data/val"
INPUT_SIZE = 224
RECOMMENDATION_CSV = "food_recommendations.csv"

# Basis data default kandungan gizi jika CSV tidak ditemukan
DEFAULT_NUTRITION_DATA = {
    "Bread": {
        "calories_per_portion": 265.0,
        "standard_portion_desc": "2 lembar (100g)",
        "carbohydrates_g": 49.0,
        "protein_g": 9.0,
        "fat_g": 3.2,
        "fiber_g": 2.7,
        "sodium_mg": 490.0,
        "health_score": 65.0,
        "nutrition_criteria": "Sumber Karbohidrat Sedang",
        "recommendation": "Baik untuk sarapan atau sumber energi cepat. Sebaiknya pilih roti gandum utuh untuk serat lebih tinggi.",
        "reason": "Mengandung karbohidrat tinggi dengan protein dan lemak moderat.",
        "dietary_notes": "Indeks glikemik relatif sedang-tinggi; batasi selai manis berlebih untuk menjaga kadar gula darah."
    },
    "nasi-goreng": {
        "calories_per_portion": 520.0,
        "standard_portion_desc": "1 piring sedang (250g)",
        "carbohydrates_g": 68.0,
        "protein_g": 14.0,
        "fat_g": 22.0,
        "fiber_g": 2.1,
        "sodium_mg": 820.0,
        "health_score": 48.0,
        "nutrition_criteria": "Tinggi Kalori & Lemak Olahan",
        "recommendation": "Konsumsi dengan porsi terkontrol. Tambahkan lalapan/sayuran segar dan pilih protein tanpa digoreng kembali.",
        "reason": "Tinggi kalori dan lemak karena proses penumisan minyak serta kecap/garam.",
        "dietary_notes": "Kandungan natrium dan lemak jenuh cukup tinggi dari minyak penggorengan; batasi frekuensi bagi penderita hipertensi/dislipidemia."
    },
    "Noodles-Pasta": {
        "calories_per_portion": 380.0,
        "standard_portion_desc": "1 mangkuk (200g)",
        "carbohydrates_g": 64.0,
        "protein_g": 12.0,
        "fat_g": 8.5,
        "fiber_g": 3.2,
        "sodium_mg": 580.0,
        "health_score": 58.0,
        "nutrition_criteria": "Padat Energi & Karbohidrat Olahan",
        "recommendation": "Porsi perlu diperhatikan. Kombinasikan dengan sayuran hijau dan sumber protein tanpa lemak agar nutrisi lebih seimbang.",
        "reason": "Dominan karbohidrat dengan lemak sedang tergantung saus atau bumbu olahan.",
        "dietary_notes": "Cepat diserap menjadi glukosa darah; disarankan porsi tidak berlebih terutama saat makan malam."
    },
    "Seafood": {
        "calories_per_portion": 210.0,
        "standard_portion_desc": "1 porsi udang/ikan/cumi (150g)",
        "carbohydrates_g": 4.5,
        "protein_g": 32.0,
        "fat_g": 6.0,
        "fiber_g": 0.5,
        "sodium_mg": 320.0,
        "health_score": 85.0,
        "nutrition_criteria": "Tinggi Protein & Rendah Lemak Jenuh",
        "recommendation": "Pilihan sangat baik untuk pemenuhan protein harian, pembentukan otot, dan diet sehat bergizi seimbang.",
        "reason": "Kandungan protein sangat tinggi dengan kadar lemak total relatif rendah dan kaya asam lemak esensial.",
        "dietary_notes": "Sangat baik untuk program diet tinggi protein dan kesehatan jantung; perhatikan cara pengolahan (hindari deep frying bertepung)."
    }
}

@st.cache_data(show_spinner=False)
def load_nutrition_database(path):
    if not os.path.exists(path):
        return DEFAULT_NUTRITION_DATA, f"File `{path}` tidak ditemukan. Menggunakan database gizi bawaan."

    data = {}
    try:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            cname = str(row["class_name"]).strip()
            data[cname] = {
                "calories_per_portion": float(row.get("calories_per_portion", row.get("health_score", 300))),
                "standard_portion_desc": str(row.get("standard_portion_desc", "1 porsi standar")),
                "carbohydrates_g": float(row.get("carbohydrates_g", 40.0)),
                "protein_g": float(row.get("protein_g", 15.0)),
                "fat_g": float(row.get("fat_g", 10.0)),
                "fiber_g": float(row.get("fiber_g", 2.0)),
                "sodium_mg": float(row.get("sodium_mg", 400.0)),
                "health_score": float(row.get("health_score", 60.0)),
                "nutrition_criteria": str(row.get("nutrition_criteria", "Gizi Standar")),
                "recommendation": str(row.get("recommendation", "")),
                "reason": str(row.get("reason", "")),
                "dietary_notes": str(row.get("dietary_notes", "")),
            }
        return data, None
    except Exception as e:
        return DEFAULT_NUTRITION_DATA, f"Gagal membaca `{path}` ({e}). Menggunakan data bawaan."

nutrition_db, db_warning = load_nutrition_database(RECOMMENDATION_CSV)

# ============================================================
# Sidebar: Konfigurasi Model
# ============================================================
st.sidebar.header(" Konfigurasi Model")
checkpoint_dir = "checkpoints"
if not os.path.exists(checkpoint_dir):
    st.sidebar.error("Folder 'checkpoints' tidak ditemukan.")
    available_models = []
else:
    available_models = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")])

if not available_models:
    st.sidebar.warning("Tidak ada checkpoint (.pth) pada folder 'checkpoints/'.")
    selected_model_file = None
else:
    default_idx = 0
    if "best_model.pth" in available_models:
        default_idx = available_models.index("best_model.pth")
    elif "ckpt_epoch30.pth" in available_models:
        default_idx = available_models.index("ckpt_epoch30.pth")

    selected_model_file = st.sidebar.selectbox(
        "Pilih Model (.pth)",
        available_models,
        index=default_idx
    )

CHECKPOINT_PATH = os.path.join(checkpoint_dir, selected_model_file) if selected_model_file else None

# Deteksi arsitektur dari parameter bobot checkpoint
ARCH = "resnet50"
if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    try:
        ckpt_temp = torch.load(CHECKPOINT_PATH, map_location="cpu")
        state_dict = ckpt_temp.get("model_state", ckpt_temp)
        first_key = next(iter(state_dict.keys()))
        if first_key.startswith("features") or "classifier" in first_key:
            ARCH = "efficientnet_b0"
        elif first_key.startswith("conv1") or first_key.startswith("layer"):
            ARCH = "resnet50"
        else:
            ARCH = ckpt_temp.get("arch", "resnet50")
    except Exception:
        ARCH = "resnet50"

if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    arch_display = "ResNet-50" if ARCH == "resnet50" else "EfficientNet-B0"
    st.sidebar.success(f"Model Terpilih: **{selected_model_file}** ({arch_display})")
else:
    st.sidebar.warning("Model belum dipilih.")

if db_warning:
    st.sidebar.warning(db_warning)

# ============================================================
# Cache Load Model
# ============================================================
@st.cache_resource(show_spinner="Memuat bobot model deep learning...")
def load_model_cached(ckpt_path, arch):
    if ckpt_path is None or not os.path.exists(ckpt_path):
        return None, []
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    classes = ckpt.get("classes", ["Bread", "Noodles-Pasta", "Seafood", "nasi-goreng"])
    model = load_checkpoint(ckpt_path, num_classes=len(classes), arch=arch)
    return model, classes

if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    model, classes = load_model_cached(CHECKPOINT_PATH, ARCH)
    transform = get_inference_transforms(INPUT_SIZE)
else:
    model, classes = None, []

# ============================================================
# Sidebar: Evaluasi Model pada Data Validasi
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader(" Evaluasi Kinerja Model")
evaluate_clicked = st.sidebar.button("Uji Model pada Data Val", use_container_width=True)

if evaluate_clicked:
    if model is None:
        st.sidebar.error("Model belum dimuat.")
    else:
        files, labels, _ = make_file_label_list(VAL_DIR)
        if len(files) == 0:
            st.sidebar.warning("Folder data/val kosong atau tidak ditemukan.")
        else:
            val_transform = transforms.Compose([
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            val_ds = FoodDataset(files, labels, val_transform)
            val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

            y_true, y_pred = [], []
            model.eval()
            with torch.no_grad():
                for imgs, labs in val_loader:
                    imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                    logits = model(imgs)
                    preds = torch.argmax(logits, dim=1)
                    y_true.extend(labs.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

            cm = confusion_matrix(y_true, y_pred)
            report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
            acc = report_dict["accuracy"]

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Hasil Evaluasi Model pada Data Uji/Validasi")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Akurasi Total", f"{acc*100:.2f}%")
            c2.metric("Macro Precision", f"{report_dict['macro avg']['precision']*100:.2f}%")
            c3.metric("Macro Recall", f"{report_dict['macro avg']['recall']*100:.2f}%")
            c4.metric("Macro F1-Score", f"{report_dict['macro avg']['f1-score']*100:.2f}%")

            col_cm, col_rep = st.columns([1, 1])
            with col_cm:
                st.markdown("**Confusion Matrix**")
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=classes, yticklabels=classes, ax=ax)
                ax.set_xlabel("Prediksi Model")
                ax.set_ylabel("Kelas Aktual")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with col_rep:
                st.markdown("**Classification Report Detail**")
                df_rep = pd.DataFrame(report_dict).transpose()
                st.dataframe(df_rep.style.format(precision=4), use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# Main App: Unggah Gambar & Analisis Kandungan Gizi
# ============================================================
st.header(" Unggah & Analisis Citra Makanan")
st.write("Unggah satu atau beberapa gambar makanan untuk mengenali kelas makanan secara otomatis dan menghitung estimasi kandungan nutrisi lengkap.")

uploaded_files = st.file_uploader(
    "Pilih satu atau beberapa gambar makanan (JPG, PNG, JPEG)",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    if model is None:
        st.error("Silakan pilih model terlebih dahulu di sidebar.")
    else:
        results = []

        for img_idx, uploaded in enumerate(uploaded_files):
            img = Image.open(uploaded).convert("RGB")
            x = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            idx = int(torch.argmax(logits, dim=1))
            label = classes[idx] if idx < len(classes) else "Unknown"
            confidence = probs[idx]

            # Ambil data kandungan gizi dasar
            nut_info = nutrition_db.get(label, DEFAULT_NUTRITION_DATA.get(label, {
                "calories_per_portion": 300.0,
                "standard_portion_desc": "1 porsi",
                "carbohydrates_g": 40.0,
                "protein_g": 10.0,
                "fat_g": 10.0,
                "fiber_g": 2.0,
                "sodium_mg": 400.0,
                "health_score": 60.0,
                "nutrition_criteria": "Standar",
                "recommendation": "-",
                "reason": "-",
                "dietary_notes": "-"
            }))

            # Input porsi interaktif
            col_img_input1, col_img_input2 = st.columns([1, 2])
            with col_img_input1:
                st.image(img, caption=uploaded.name, use_container_width=True)
            with col_img_input2:
                st.markdown(f"### **{uploaded.name}** → Terdeteksi: `{label}` ({confidence*100:.2f}%)")
                st.markdown(f"**Porsi Standar Referensi:** {nut_info.get('standard_portion_desc', '1 porsi')}")
                portion_pct = st.slider(
                    f"Sesuaikan Persentase Porsi untuk {uploaded.name} (%)",
                    min_value=25,
                    max_value=300,
                    value=100,
                    step=25,
                    key=f"portion_{img_idx}_{uploaded.name}"
                )

            scale = portion_pct / 100.0

            # Estimasi kandungan gizi terhitung berdasarkan skala porsi
            calculated_cal = nut_info["calories_per_portion"] * scale
            calculated_carbs = nut_info["carbohydrates_g"] * scale
            calculated_protein = nut_info["protein_g"] * scale
            calculated_fat = nut_info["fat_g"] * scale
            calculated_fiber = nut_info["fiber_g"] * scale
            calculated_sodium = nut_info["sodium_mg"] * scale

            results.append({
                "file_name": uploaded.name,
                "image": img,
                "label": label,
                "confidence": confidence,
                "probs": probs,
                "logits": logits,
                "portion_pct": portion_pct,
                "scale": scale,
                "nut_info": nut_info,
                "cal": calculated_cal,
                "carbs": calculated_carbs,
                "protein": calculated_protein,
                "fat": calculated_fat,
                "fiber": calculated_fiber,
                "sodium": calculated_sodium,
                "health_score": nut_info["health_score"]
            })

            st.markdown("---")

        # ============================================================
        # Ringkasan & Tabel Kandungan Gizi
        # ============================================================
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader(" Ringkasan Analisis Kandungan Gizi")

        summary_rows = []
        for r in results:
            summary_rows.append({
                "Nama Gambar": r["file_name"],
                "Kelas Terdeteksi": r["label"],
                "Confidence": f"{r['confidence']*100:.2f}%",
                "Porsi Terpilih": f"{r['portion_pct']}%",
                "Kalori (kkal)": f"{r['cal']:.1f}",
                "Karbohidrat (g)": f"{r['carbs']:.1f}",
                "Protein (g)": f"{r['protein']:.1f}",
                "Lemak (g)": f"{r['fat']:.1f}",
                "Serat (g)": f"{r['fiber']:.1f}",
                "Natrium (mg)": f"{r['sodium']:.0f}",
                "Kriteria Gizi": r["nut_info"]["nutrition_criteria"],
                "Skor Kesehatan": f"{r['health_score']:.0f}/100"
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ============================================================
        # Komparasi & Rekomendasi Antar Makanan (Jika >= 2 gambar)
        # ============================================================
        if len(results) >= 2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader(" Komparasi & Rekomendasi Pilihan Makanan")
            goal = st.selectbox(
                "Pilih Kriteria Rekomendasi:",
                [
                    "Gizi Paling Seimbang (Skor Kesehatan Tertinggi)",
                    "Kandungan Protein Tertinggi",
                    "Kalori Paling Rendah (Diet Defisit Kalori)",
                    "Kandungan Lemak Paling Rendah"
                ]
            )

            if goal == "Gizi Paling Seimbang (Skor Kesehatan Tertinggi)":
                best = max(results, key=lambda x: x["health_score"])
                reason_txt = f"Memiliki skor kesehatan tertinggi ({best['health_score']:.0f}/100) dengan kriteria: **{best['nut_info']['nutrition_criteria']}**. {best['nut_info']['recommendation']}"
            elif goal == "Kandungan Protein Tertinggi":
                best = max(results, key=lambda x: x["protein"])
                reason_txt = f"Menyediakan protein tertinggi ({best['protein']:.1f} g) yang sangat baik untuk regenerasi sel tubuh dan rasa kenyang lebih lama. {best['nut_info']['recommendation']}"
            elif goal == "Kalori Paling Rendah (Diet Defisit Kalori)":
                best = min(results, key=lambda x: x["cal"])
                reason_txt = f"Memberikan asupan energi paling efisien ({best['cal']:.1f} kkal) untuk menjaga berat badan. {best['nut_info']['recommendation']}"
            else:
                best = min(results, key=lambda x: x["fat"])
                reason_txt = f"Memiliki total lemak terendah ({best['fat']:.1f} g) untuk meminimalisir risiko penumpukan kolesterol/lemak jenuh. {best['nut_info']['recommendation']}"

            st.success(f" **Pilihan Terbaik:** **{best['file_name']}** (`{best['label']}`) — {reason_txt}")

            # Visualisasi Komparasi Bar Chart
            fig, ax = plt.subplots(figsize=(10, 4))
            labels = [f"{r['file_name']}\n({r['label']})" for r in results]
            x = np.arange(len(labels))
            width = 0.2

            ax.bar(x - width*1.5, [r["cal"] for r in results], width, label="Kalori (kkal)", color="#3b82f6")
            ax.bar(x - width*0.5, [r["carbs"] for r in results], width, label="Karbo (g)", color="#eab308")
            ax.bar(x + width*0.5, [r["protein"] for r in results], width, label="Protein (g)", color="#22c55e")
            ax.bar(x + width*1.5, [r["fat"] for r in results], width, label="Lemak (g)", color="#ef4444")

            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel("Nilai Nutrisi")
            ax.set_title("Perbandingan Nilai Kandungan Nutrisi per Gambar")
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown('</div>', unsafe_allow_html=True)

        # ============================================================
        # Detail Kandungan Makronutrisi Tiap Gambar
        # ============================================================
        st.subheader(" Rincian Kandungan Gizi Masing-Masing Menu")
        cols = st.columns(min(3, len(results)))
        for idx, r in enumerate(results):
            with cols[idx % len(cols)]:
                st.markdown(f"#### **{r['file_name']}**")
                st.image(r["image"], use_container_width=True)
                st.markdown(f"**Kelas:** `{r['label']}` (Keyakinan: {r['confidence']*100:.1f}%)")
                st.markdown(f"**Kriteria:** <span style='background:#dbeafe;color:#1e40af;padding:3px 8px;border-radius:6px;font-size:12px;font-weight:600;'>{r['nut_info']['nutrition_criteria']}</span>", unsafe_allow_html=True)

                # Metrik Card Makronutrisi
                m1, m2 = st.columns(2)
                m1.metric("Energi Total", f"{r['cal']:.0f} kkal")
                m2.metric("Protein", f"{r['protein']:.1f} g")
                m3, m4 = st.columns(2)
                m3.metric("Karbohidrat", f"{r['carbs']:.1f} g")
                m4.metric("Lemak Total", f"{r['fat']:.1f} g")

                # Donut Chart Makronutrisi
                macro_labels = ["Karbohidrat", "Protein", "Lemak"]
                macro_values = [r["carbs"] * 4, r["protein"] * 4, r["fat"] * 9]  # kkal per makronutrisi
                fig_pie, ax_pie = plt.subplots(figsize=(3.5, 3.5))
                colors = ["#eab308", "#22c55e", "#ef4444"]
                ax_pie.pie(macro_values, labels=macro_labels, colors=colors, autopct='%1.0f%%', startangle=90,
                           wedgeprops=dict(width=0.4, edgecolor='white'))
                ax_pie.set_title("Distribusi Energi Makronutrisi", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig_pie)
                plt.close(fig_pie)

                # Catatan Diet dan Rekomendasi
                st.info(f" **Saran Konsumsi:** {r['nut_info']['recommendation']}")
                st.caption(f" **Catatan Klinis/Diet:** {r['nut_info']['dietary_notes']}")

        # ============================================================
        # Penjelasan Landasan Ilmiah & Formula Matematis (Sesuai Metode)
        # ============================================================
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader(" Landasan Metodologi Ilmiah & Implementasi Algoritma")

        with st.expander("1. Tahap Klasifikasi Citra (Deep Convolutional Neural Network & Softmax)", expanded=False):
            st.markdown(
                """
                Model deep learning (**ResNet-50 / EfficientNet-B0**) mengekstrak fitur spasial citra makanan melalui lapisan konvolusi berulang.
                Vektor representasi fitur diteruskan ke lapisan *Fully Connected* (FC) yang menghasilkan nilai logit non-probabilistik $z_i$.
                
                Untuk mengonversi logit menjadi distribusi probabilitas kelas terstandarisasi, diterapkan fungsi aktivasi **Softmax**:
                """
            )
            st.latex(r"P(y = i \mid \mathbf{x}) = \sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}")
            st.markdown(
                """
                Di mana:
                - $C$ = Total jumlah kelas makanan dalam dataset ($C=4$).
                - $z_i$ = Nilai logit untuk kelas ke-$i$.
                - $P(y = i \mid \mathbf{x})$ = Tingkat kepercayaan (*Confidence Score*) bahwa citra merupakan kelas makanan $i$.
                """
            )

        with st.expander("2. Tahap Estimasi Kandungan Gizi & Kalori (Sistem Faktor Atwater)", expanded=False):
            st.markdown(
                """
                Setelah kelas makanan berhasil diklasifikasikan, estimasi total nilai energi (kalori) dihitung menggunakan **Sistem Faktor Umum Atwater** (*Atwater General Factor System*) yang diakui secara internasional oleh FAO/WHO dan Kementerian Kesehatan RI:
                """
            )
            st.latex(r"\text{Energi Total (kkal)} = \left[ (4 \times \text{Karbohidrat (g)}) + (4 \times \text{Protein (g)}) + (9 \times \text{Lemak (g)}) \right] \times \frac{\text{Porsi}(\%)}{100}")
            st.markdown(
                """
                Koefisien fisiologis energi Atwater:
                - **1 gram Karbohidrat** = 4 kkal energi metabolis.
                - **1 gram Protein** = 4 kkal energi metabolis.
                - **1 gram Lemak** = 9 kkal energi metabolis.
                
                Dengan pendekatan ini, sistem mengintegrasikan **Computer Vision** untuk identifikasi jenis pangan secara objektif, dan **Nutritional Modeling** untuk perhitungan gizi berbasis standar empiris.
                """
            )
