import os
import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Belajar Klasifikasi Buah",
    page_icon="🍎",
    layout="centered"
)

# Load Model secara aman (relatif terhadap file ini)
model_path = os.path.join(os.path.dirname(__file__), "model_klasifikasi_buah.joblib")
model = joblib.load(model_path)

st.title("🍎 🍊 🥭 Klasifikasi Buah")
st.markdown("Aplikasi machine learning classification untuk memprediksi jenis buah (**Apel**, **Jeruk**, atau **Mangga**) berdasarkan karakteristik fisik dan asalnya.")

st.subheader("📋 Input Karakteristik Buah")

# Input Numerik (disesuaikan dengan batas data aktual pada dataset)
col1, col2 = st.columns(2)
with col1:
    diameter = st.slider("Diameter (cm)", min_value=4.0, max_value=10.0, value=6.5, step=0.1)
    tebal_kulit = st.slider("Tebal Kulit (cm)", min_value=0.2, max_value=1.0, value=0.58, step=0.01)

with col2:
    berat = st.slider("Berat (gram)", min_value=50.0, max_value=400.0, value=180.0, step=1.0)
    kadar_gula = st.slider("Kadar Gula (Brix / %)", min_value=7.0, max_value=20.0, value=12.5, step=0.1)

# Input Kategorikal
daftar_asal = ["Bandung", "Banyuwangi", "Batu", "Garut", "Indramayu", "Magelang", "Malang", "Pasuruan", "Probolinggo"]
daftar_warna = ["hijau", "hijau kekuningan", "kuning", "kuning kemerahan", "merah", "oranye", "oranye kekuningan"]
daftar_musim = ["kemarau", "hujan"]

asal_daerah = st.pills("Asal Daerah", daftar_asal, default="Garut")
warna = st.pills("Warna", daftar_warna, default="hijau")
musim_panen = st.pills("Musim Panen", daftar_musim, default="hujan")

# Tombol Prediksi
if st.button("🔮 Prediksi Jenis Buah", type="primary", use_container_width=True):
    # Validasi jika pengguna tidak sengaja meng-unselect pill
    if not asal_daerah or not warna or not musim_panen:
        st.warning("⚠️ Mohon pastikan Asal Daerah, Warna, dan Musim Panen telah dipilih!")
    else:
        data_baru = pd.DataFrame(
            [[diameter, berat, tebal_kulit, kadar_gula, asal_daerah, warna, musim_panen]],
            columns=["diameter", "berat", "tebal_kulit", "kadar_gula", "asal_daerah", "warna", "musim_panen"]
        )

        prediksi = model.predict(data_baru)[0]
        probabilitas = model.predict_proba(data_baru)[0]
        presentase = max(probabilitas) * 100

        # Mapping emoji untuk buah
        emoji_buah = {
            "Apel": "🍎",
            "Jeruk": "🍊",
            "Mangga": "🥭"
        }
        icon = emoji_buah.get(prediksi, "🍏")

        st.success(f"### Model memprediksi buah ini adalah: **{icon} {prediksi}**")
        st.info(f"Tingkat keyakinan model: **{presentase:.2f}%**")

        # Tampilkan rincian probabilitas untuk setiap buah
        st.write("##### Rincian Probabilitas Tiap Buah:")
        prob_df = pd.DataFrame({
            "Jenis Buah": [f"{emoji_buah.get(c, '')} {c}" for c in model.classes_],
            "Peluang": [f"{p * 100:.2f}%" for p in probabilitas]
        })
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

        st.balloons()

st.divider()
st.caption("Dibuat dengan ❤️ oleh **Ananda Nauval**")