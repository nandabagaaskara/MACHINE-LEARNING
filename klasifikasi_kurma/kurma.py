import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Aplikasi Menebak jenis kurma",
    page_icon="🫘"
)

model = joblib.load("kurma.joblib")

st.title("🫘 Belajar Klasifikasi Kurma")
st.markdown("Aplikasi Machine Learning untuk Mengklasifikasi Jenis Kurma")

ukuran_cm = st.slider("Ukuran(cm)", 1.62, 6.11, 4.05)
berat_g = st.slider("Berat(gram)", 3.43, 20.10, 11.54)
kekerasan_N = st.slider("Kekerasan(N)", 2.0, 89.94, 58.27)
kadar_gula_brix = st.slider("Kadar Gula(Brix)", 13.8, 82.81, 61.67)
kadar_air_pct = st.slider("Kadar Air(%)", 10.05, 79.99, 66.17)

varietas = st.pills("Varietas", ["Deglet Noor", "Zahidi", "Medjool", "Ajwa"], default="Deglet Noor")
tingkat_warna = st.pills("Tingkat Warna", ["Hitam", "Coklat Tua", "Coklat Muda", "Kuning"], default="Hitam")
grade_cacat = st.pills("Cacat", ["Tidak Ada", "Ringan", "Sedang", "Berat"], default="Ringan")

if st.button("Prediksi", type="primary"):
    dataframe = pd.DataFrame(
        [[ukuran_cm, berat_g, kekerasan_N, kadar_gula_brix, kadar_air_pct,
          varietas, tingkat_warna, grade_cacat]],
        columns=["ukuran_cm", "berat_g", "kekerasan_N", "kadar_gula_brix",
                 "kadar_air_pct", "varietas", "tingkat_warna", "grade_cacat"]
    )

    prediksi = model.predict(dataframe)[0]
    presentase = max(model.predict_proba(dataframe)[0])
    st.success(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
    st.snow()

st.divider()
st.caption("Dibuat oleh **Ananda Nauval**")