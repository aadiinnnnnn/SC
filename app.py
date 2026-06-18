import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Harga Rumah", page_icon="🏡", layout="centered")

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_data():
    model = joblib.load('model_rumah_jakarta.pkl')
    cities = joblib.load('cities.pkl')
    districts = joblib.load('districts.pkl')
    return model, cities, districts

model, cities, districts = load_data()

# --- HEADER APLIKASI ---
st.title("🏡 Sistem Prediksi Harga Rumah Perkotaan")
st.markdown("Aplikasi estimasi harga rumah di kawasan Jakarta berbasis **Ensemble Regression (Random Forest + XGBoost)**.")
st.divider()

# --- SIDEBAR: INPUT PENGGUNA ---
st.sidebar.header("Data Properti")

city = st.sidebar.selectbox("Pilih Kota", cities)
# Filter distrik secara berurutan sesuai abjad
district = st.sidebar.selectbox("Pilih Kecamatan/Distrik", sorted(districts))

building_area = st.sidebar.number_input("Luas Bangunan (m2)", min_value=10.0, max_value=2000.0, value=150.0, step=10.0)
land_area = st.sidebar.number_input("Luas Tanah (m2)", min_value=10.0, max_value=2000.0, value=120.0, step=10.0)
bed_rooms = st.sidebar.number_input("Jumlah Kamar Tidur", min_value=1, max_value=20, value=3)
bath_rooms = st.sidebar.number_input("Jumlah Kamar Mandi", min_value=1, max_value=20, value=2)
carport = st.sidebar.number_input("Kapasitas Carport (Mobil)", min_value=0, max_value=10, value=1)

# --- TOMBOL PREDIKSI ---
if st.sidebar.button("Estimasi Harga", type="primary"):
    
    # 1. Menyiapkan Data Input
    input_data = pd.DataFrame({
        'building_area': [building_area],
        'land_area': [land_area],
        'bed_rooms': [bed_rooms],
        'bath_rooms': [bath_rooms],
        'carport': [carport],
        'city': [city],
        'district': [district]
    })
    
    # 2. Proses Prediksi Harga
    prediksi = model.predict(input_data)[0]
    
    # Menampilkan Hasil Format Rupiah
    st.success("### Estimasi Nilai Properti:")
    if prediksi >= 1e9:
        st.write(f"## Rp {prediksi/1e9:,.2f} Miliar")
    else:
        st.write(f"## Rp {prediksi/1e6:,.2f} Juta") 
        
    st.divider()
    
    # 3. PENJELASAN FAKTOR UTAMA (Deliverable dari Slide)
    st.subheader("📊 Penjelasan Faktor Utama (Feature Importance)")
    st.markdown("Grafik ini menganalisis bobot fitur mana yang paling memengaruhi sistem dalam menentukan harga rumah.")
    
    try:
        # Ekstraksi komponen dari Pipeline
        preprocessor = model.named_steps['preprocessor']
        # Mengambil model Random Forest yang terbungkus di dalam TransformedTargetRegressor dan VotingRegressor
        rf_model = model.named_steps['model'].regressor_.estimators_[0]
        
        # Dapatkan nama fitur asli yang sudah di-OneHotEncode
        feature_names = preprocessor.get_feature_names_out()
        # Merapikan label agar enak dibaca di grafik
        clean_features = []
        for f in feature_names:
            f = f.replace('num__', '')
            f = f.replace('cat__city_', 'Kota: ')
            f = f.replace('cat__district_', 'Kec: ')
            clean_features.append(f)
            
        importances = rf_model.feature_importances_
        
        # Ambil 10 fitur paling berpengaruh saja agar grafik tidak kepenuhan
        indices = np.argsort(importances)[-10:] 
        
        # Plotting Grafik Horizontal Bar
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(len(indices)), importances[indices], color='#2E86C1')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([clean_features[i] for i in indices])
        ax.set_xlabel("Tingkat Signifikansi Pengaruh")
        ax.set_title("10 Faktor Paling Berpengaruh Terhadap Prediksi Harga")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Tampilkan visualisasi di Streamlit
        st.pyplot(fig)
        
    except Exception as e:
        st.warning("Visualisasi faktor utama tidak dapat dimuat secara teknis.")