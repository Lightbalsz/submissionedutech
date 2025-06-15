import streamlit as st
import pandas as pd
from prediction import data_preprocessing, prediction

st.set_page_config(page_title="Jaya Jaya Dropout Predictor", page_icon=":school:", layout="wide")
st.title(":school: Jaya Jaya Institute Dropout Prediction :school:")

# Form input data
st.header("Personal Information", divider="rainbow")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Laki-laki", "Perempuan"], index=0)
with col2:
    age_at_enrollment = st.number_input("Age at Enrollment", min_value=15, max_value=80, value=22)

col1, col2, col3 = st.columns(3)
with col1:
    debtor = st.selectbox("Debtor", ["Tidak", "Ya"], index=0)
with col2:
    scholarship_holder = st.selectbox("Scholarship Holder", ["Tidak", "Ya"], index=0)
with col3:
    tuition_fees = st.selectbox("Tuition Fees Up To Date", ["Tidak", "Ya"], index=0)

st.header("Curricular Units 1st Semester Information", divider="rainbow")
col1, col2, col3, col4 = st.columns(4)
with col1:
    curr_1st_enrolled = st.number_input("Enrolled (0 - 30)", 0, 30, 0)
with col2:
    curr_1st_eval = st.number_input("Evaluations (0 - 50)", 0, 50, 0)
with col3:
    curr_1st_approved = st.number_input("Approved (0 - 30)", 0, 30, 0)
with col4:
    curr_1st_grade = st.number_input("Grade (0 - 20)", 0.0, 20.0, 0.0)

st.header("Curricular Units 2nd Semester Information", divider="rainbow")
col1, col2, col3, col4 = st.columns(4)
with col1:
    curr_2nd_enrolled = st.number_input("Enrolled (0 - 30)", 0, 30, 0, key="2nd_enrolled")
with col2:
    curr_2nd_eval = st.number_input("Evaluations (0 - 50)", 0, 50, 0, key="2nd_eval")
with col3:
    curr_2nd_approved = st.number_input("Approved (0 - 30)", 0, 30, 0, key="2nd_approved")
with col4:
    curr_2nd_grade = st.number_input("Grade (0 - 20)", 0.0, 20.0, 0.0, key="2nd_grade")

# Lihat input
input_df = pd.DataFrame([{
    'Gender': gender,
    'Age_at_enrollment': age_at_enrollment,
    'Debtor': debtor,
    'Scholarship_holder': scholarship_holder,
    'Tuition_fees_up_to_date': tuition_fees,
    'Curricular_units_1st_sem_enrolled': curr_1st_enrolled,
    'Curricular_units_1st_sem_evaluations': curr_1st_eval,
    'Curricular_units_1st_sem_approved': curr_1st_approved,
    'Curricular_units_1st_sem_grade': curr_1st_grade,
    'Curricular_units_2nd_sem_enrolled': curr_2nd_enrolled,
    'Curricular_units_2nd_sem_evaluations': curr_2nd_eval,
    'Curricular_units_2nd_sem_approved': curr_2nd_approved,
    'Curricular_units_2nd_sem_grade': curr_2nd_grade
}])

with st.expander("Lihat data input"):
    st.dataframe(input_df)

# Susun sesuai urutan untuk model
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

# Prediksi
if st.button("🔮 Predict Dropout"):
    data_preprocessed = data_preprocessing(data_input)
    result = prediction(data_preprocessed, data_input)

    with st.expander("Lihat data setelah preprocessing"):
        st.dataframe(data_preprocessed)

    if result == "Graduate":
        st.success("🎓 Selamat! Siswa diprediksi akan LULUS.")
    else:
        st.error("⚠️ Siswa berpotensi mengalami DROPOUT. Perlu perhatian lebih.{result}")
