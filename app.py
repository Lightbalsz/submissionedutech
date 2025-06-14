import streamlit as st
import pandas as pd
import numpy as np
from prediction import data_preprocessing, prediction

st.set_page_config(page_title="Jaya Jaya Dropout Predictor", page_icon=":school:", layout="wide")

# Judul Aplikasi
st.title(":school: Jaya Jaya Institute Dropout Prediction :school:")

# Form Input Data
st.header("Personal Information", divider="rainbow")
col1, col2 = st.columns(2)

raw_data = pd.DataFrame()

with col1:
    gender = st.selectbox(label="Gender", options=["Laki-laki", "Perempuan"], index=0)
    raw_data["Gender"] = [gender]

with col2:
    age_at_enrollment = st.number_input("Age at Enrollment", value=22)
    raw_data["Age_at_enrollment"] = [age_at_enrollment]

col1, col2, col3 = st.columns(3)

with col1:
    debtor = st.selectbox(label="Debtor", options=["Tidak", "Ya"], index=0)
    raw_data["Debtor"] = [debtor]

with col2:
    scholarship_holder = st.selectbox(label="Scholarship Holder", options=["Tidak", "Ya"], index=0)
    raw_data["Scholarship_holder"] = [scholarship_holder]

with col3:
    tuition_fees = st.selectbox(label="Tuition Fees Up To Date", options=["Tidak", "Ya"], index=0)
    raw_data["Tuition_fees_up_to_date"] = [tuition_fees]

# Informasi Semester 1
st.header("Curricular Units 1st Semester Information", divider="rainbow")
col1, col2, col3, col4 = st.columns(4)

with col1:
    curr_1st_enrolled = int(st.number_input("Enrolled (0 - 30)", 0, 30, 20, key="1st sem enrolled"))
    raw_data["Curricular_units_1st_sem_enrolled"] = [curr_1st_enrolled]

with col2:
    curr_1st_eval = int(st.number_input("Evaluations (0 - 50)", 0, 50, 35, key="1st sem evaluations"))
    raw_data["Curricular_units_1st_sem_evaluations"] = [curr_1st_eval]

with col3:
    curr_1st_approved = int(st.number_input("Approved (0 - 30)", 0, 30, 20, key="1st sem approved"))
    raw_data["Curricular_units_1st_sem_approved"] = [curr_1st_approved]

with col4:
    curr_1st_grade = float(st.number_input("Grade (0 - 20)", 0.0, 20.0, 17.0, key="1st sem grade"))
    raw_data["Curricular_units_1st_sem_grade"] = [curr_1st_grade]

# Informasi Semester 2
st.header("Curricular Units 2nd Semester Information", divider="rainbow")
col1, col2, col3, col4 = st.columns(4)

with col1:
    curr_2nd_enrolled = int(st.number_input("Enrolled (0 - 30)", 0, 30, 20, key="2nd sem enrolled"))
    raw_data["Curricular_units_2nd_sem_enrolled"] = [curr_2nd_enrolled]

with col2:
    curr_2nd_eval = int(st.number_input("Evaluations (0 - 50)", 0, 50, 35, key="2nd sem evaluations"))
    raw_data["Curricular_units_2nd_sem_evaluations"] = [curr_2nd_eval]

with col3:
    curr_2nd_approved = int(st.number_input("Approved (0 - 30)", 0, 30, 20, key="2nd sem approved"))
    raw_data["Curricular_units_2nd_sem_approved"] = [curr_2nd_approved]

with col4:
    curr_2nd_grade = float(st.number_input("Grade (0 - 20)", 0.0, 20.0, 17.0, key="2nd sem grade"))
    raw_data["Curricular_units_2nd_sem_grade"] = [curr_2nd_grade]

# Tampilkan Data Masukan
with st.expander("Lihat data input"):
    st.dataframe(data=raw_data, width=900)

# Siapkan data untuk prediksi
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
    data_preprocessed = data_preprocessing(data_input)
    result = prediction(data_preprocessed)

    with st.expander("Lihat data setelah preprocessing"):
        st.dataframe(data=data_preprocessed)

    if result == "Graduate":
        st.success("🎓 Selamat! Siswa diprediksi akan LULUS.")
    elif result == "Dropout":
        st.error("⚠️ Siswa berpotensi mengalami DROPOUT. Perlu perhatian lebih.")
    else:
        st.info("📘 Siswa masih TERDAFTAR.")

