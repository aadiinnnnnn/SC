<div align="center">

# 🏡 Sistem Prediksi Harga Rumah Jakarta

*Aplikasi web interaktif untuk mengestimasi harga properti di DKI Jakarta menggunakan Machine Learning Ensemble.*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-179331?style=for-the-badge&logo=xgboost&logoColor=white)

</div>

---

## 📖 Tentang Proyek
Proyek ini adalah sebuah aplikasi interaktif berbasis web yang memprediksi harga rumah di kawasan Jakarta. Menggunakan kombinasi algoritma **Random Forest** dan **XGBoost** (Voting Regressor), aplikasi ini mampu memberikan estimasi harga berdasarkan berbagai spesifikasi properti yang diinputkan oleh pengguna.

> **Demo Aplikasi:** *(Tambahkan link aplikasi Anda di sini jika sudah di-deploy, misal: [Klik di sini untuk mencoba!](https://tautan-streamlit-anda.app))*

---

## ✨ Fitur Utama
- **🎯 Estimasi Harga Akurat**: Memprediksi harga rumah secara *real-time* berdasarkan parameter Kota, Kecamatan, Luas Tanah, Luas Bangunan, Jumlah Kamar Tidur & Mandi, serta Kapasitas Carport.
- **📊 Visualisasi *Feature Importance***: Dilengkapi dengan grafik interaktif yang menampilkan 10 faktor utama yang paling memengaruhi estimasi harga properti.
- **🛡️ *Robust Preprocessing***: Sistem secara otomatis memfilter data anomali (*outliers*) dan menerapkan transformasi logaritmik (`TransformedTargetRegressor`) untuk menjaga stabilitas prediksi.

---

## 📸 *Screenshot* Aplikasi
<div align="center">
  <img src="https://via.placeholder.com/800x400.png?text=Ganti+Gambar+Ini+Dengan+Screenshot+Aplikasi+Anda" alt="Screenshot Aplikasi">
</div>

---

## 🛠️ Teknologi yang Digunakan
| Kategori | Teknologi/Library | Deskripsi |
| :--- | :--- | :--- |
| **Core** | `Python` | Bahasa pemrograman utama. |
| **Frontend** | `Streamlit` | Membangun UI/UX web app yang interaktif. |
| **Machine Learning**| `Scikit-Learn`, `XGBoost` | Pembuatan model prediktif[cite: 2]. |
| **Data Processing** | `Pandas`, `NumPy` | Manipulasi dan analisis data[cite: 2]. |
| **Visualization** | `Matplotlib` | Pembuatan grafik dan plot visual[cite: 2]. |

---

## 📂 Struktur Repositori
```text
aadiinnnnnn/sc
├── app.py                   # Script utama aplikasi web Streamlit
├── train_model.py           # Script pipeline untuk preprocessing dan training model
├── jakarta_house.csv        # Dataset mentah harga rumah (pastikan ada di direktori Anda)
├── model_rumah_jakarta.pkl  # File model ML yang sudah dilatih (ter-generate dari train_model.py)
├── cities.pkl               # Data dropdown list kota (ter-generate dari train_model.py)
├── districts.pkl            # Data dropdown list kecamatan (ter-generate dari train_model.py)
└── README.md                # Dokumentasi proyek
