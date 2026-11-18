import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
	page_title = "Belajar Klasifikasi Kentang",
	page_icon = "🥔"
)
	

model = joblib.load("kentang.joblib")

st.title("🥔 Belajar Klasifikasi Kentang")
st.markdown("Aplikasi Machine Learning untuk Mengklasifikasi Jenis Kentang")

berat_gram = st.slider("Berat (gram)", 50, 315, 150)
diameter_cm = st.slider("Diameter (cm)", 4.0, 9.7, 6.0)
kadar_air_persen = st.slider("Kadar Air (%)", 65.2, 80.0, 70.0)

jenis_kentang = st.pills("Jenis Kentang", ["Russet", "Atlantik", "Kentang Merah", "Granola"], default="Russet")
metode_panen = st.pills("Metode Panen", ["Manual", "Mesin"], default="Mesin")
kondisi_penyimpanan = st.pills("Kondisi Penyimpanan", ["Cold Storage", "Suhu Ruang", "Gudang"], default="Gudang")
ada_tunas = st.pills("Ada Tunas", ["Tidak", "Ya"], default="Tidak")
tingkat_kematangan = st.pills("Kematangan Kentang", ["Belum Matang", "Matang", "Sangat Matang"], default="Sangat Matang")

if st.button("Prediksi", type="primary"):
    dataframe = pd.DataFrame(
        [[berat_gram, diameter_cm, kadar_air_persen, jenis_kentang,
          metode_panen, kondisi_penyimpanan, ada_tunas, tingkat_kematangan]],
        columns=["berat_gram", "diameter_cm", "kadar_air_persen", "jenis_kentang",
                 "metode_panen", "kondisi_penyimpanan", "ada_tunas", "tingkat_kematangan"]
    )
    prediksi = model.predict(dataframe)[0]
    presentase = max(model.predict_proba(dataframe)[0])
    st.success(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
    st.balloons()

st.divider()
st.caption("Dibuat oleh **Ananda Nauval**")



