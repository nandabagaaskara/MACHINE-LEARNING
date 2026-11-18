import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title = "Aplikasi Klasifikasi Kentang",
    page_icon = ""
)

model = joblib.load("kentang_baru.joblib")

st.title("Klasifikasi Kentang")
st.markdown("Aplikasi Klasifikasi Kentang")

berat = st.slider("Berat (Gr)", 60.0, 179.3, 309.9)
diameter = st.slider("Diameter (Cm)", 4.0, 6.3, 8.9)
kadar_air = st.slider("Kadar Air", 65.2, 76.7, 88.0)
jenis_kentang = st.pills("Jenis Kentang", ["Russet", "Granola", "Kentang Merah"], default = "Russet")
metode_panen = st.pills("Metode Panen", ["Manual", "Mesin"], default = "Manual")
kondisi_peyimpanan = st.pills("Kondisi Penyimpanan", ["Suhu Ruang", "Gudang", "Cold Storage"], default = "Gudang")
ada_tunas = st.pills("Ada Tunas", ["Ya", "Tidak"], default = "Ya")
tingkat_kematangan = st.pills("Tingkat Kematangan", ["Belum Matang", "Matang", "Sangat Matang"], default = "Matang")

if st.button("prediksi", type = "primary"):
    data_baru = pd.DataFrame([[220, 6.5, 75, "Granola", "Manual","Gudang", "Tidak", "Sangat Matang"]],
                         columns=["berat", "diameter", "kadar_air", "jenis_kentang", "metode_panen", "kondisi_penyimpanan", "ada_tunas", "tingkat_kematangan"])
    prediksi = model.predict(data_baru)[0]
    presentase = max(model.predict_proba(data_baru)[0])
    st.success(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan {presentase*100:.2f}%")
    st.snow()

st.divider()
st.caption("Dibuat oleh **Nopal Ganteng**")