import streamlit as st
import pandas as pd
from prediction import data_preprocessing, prediction

st.set_page_config(page_title="Jaya Jaya Dropout Predictor", page_icon=":school:", layout="wide")

st.title(":school: Jaya Jaya Institute Dropout Prediction :school:")

# Form Input Data
st.header("Personal Information", divider="rainbow")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Laki-laki", "Perempuan"])
    age_at_enrollment = st.number_input("Age at Enrollment", value=22)

with col2:
    debtor = st.selectbox("Debtor", ["Tidak", "Ya"])
    scholarship_holder = st.selectbox("Scholarship Holder", ["Tidak", "Ya"])
    tuition_fees = st.selectbox("Tuition Fees Up To Date", ["Tidak", "Ya"])

# Informasi Semester 1
st.header("Curricular Units 1st Semester Information", divider="rainbow")
col1, col2, col3, col4 = st.columns(4)
curr_1st_enrolled = col1.number_input("Enrolled (0 - 30)", 0, 30, 20)
curr_1st_eval = col2.number_input("Evaluations (0 - 50)", 0, 50, 35)
curr_1st_approved = col3.number_input("Approved (0 - 30)", 0, 30, 20)
curr_1st_grade = col4.number_input("Grade (0 - 20)", 0.0, 20.0, 17.0)

# Informasi Semester 2
st.header("Curricular Units 2nd Semester Information", divider="rainbow")
col1, col2, col3, col4 = st.columns(4)
curr_2nd_enrolled = col1.number_input("Enrolled (0 - 30)", 0, 30, 20)
curr_2nd_eval = col2.number_input("Evaluations (0 - 50)", 0, 50, 35)
curr_2nd_approved = col3.number_input("Approved (0 - 30)", 0, 30, 20)
curr_2nd_grade = col4.number_input("Grade (0 - 20)", 0.0, 20.0, 17.0)

# Siapkan data input
data_input = [
    curr_1st_enrolled,
    curr_2nd_enrolled,
    curr_1st_approved,
    curr_2nd_approved,
    curr_1st_grade,
    curr_2nd_grade,
    curr_1st_eval,
    curr_2nd_eval,
    age_at_enrollment,
    gender,
    debtor,
    tuition_fees,
    scholarship_holder
]

# Tombol prediksi
if st.button("🔮 Predict Dropout"):
    df_scaled = data_preprocessing(data_input)
    result = prediction(df_scaled)

    st.subheader("Hasil Prediksi:")
    if result == "Graduate":
        st.success("🎓 Selamat! Siswa diprediksi akan LULUS.")
    elif result == "Dropout":
        st.error("⚠️ Siswa berpotensi mengalami DROPOUT. Perlu perhatian lebih.")
    else:
        st.info("📘 Siswa masih TERDAFTAR.")

    with st.expander("Data setelah preprocessing"):
        st.dataframe(df_scaled)