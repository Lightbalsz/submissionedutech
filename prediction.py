import numpy as np
import joblib
import pandas as pd

# Load model dan scaler
model = joblib.load("model_dropout.pkl")
scaler = joblib.load("scaler.pkl")

# Urutan kolom yang dipakai oleh model
FEATURE_ORDER = [
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_1st_sem_evaluations',
    'Curricular_units_2nd_sem_evaluations',
    'Age_at_enrollment',
    'Gender',
    'Debtor',
    'Tuition_fees_up_to_date',
    'Scholarship_holder'
]

# Mapping nilai kategorikal ke biner
def map_input(gender, debtor, tuition_fees, scholarship_holder):
    return [
        1 if gender == "Perempuan" else 0,
        1 if debtor == "Ya" else 0,
        1 if tuition_fees == "Ya" else 0,
        1 if scholarship_holder == "Ya" else 0
    ]

# Fungsi preprocessing
def data_preprocessing(input_array):
    # Pecah input menjadi numerik & kategorikal
    numeric_part = np.array(input_array[:9], dtype=float)  # 9 fitur numerik
    categorical_part = np.array(map_input(*input_array[9:]), dtype=int)  # 4 fitur kategorikal

    # Gabungkan dan buat dataframe dengan nama kolom
    full_input = np.concatenate([numeric_part, categorical_part])
    df_input = pd.DataFrame([full_input], columns=FEATURE_ORDER)

    # Scaling
    df_scaled = pd.DataFrame(scaler.transform(df_input), columns=FEATURE_ORDER)
    return df_scaled

# Fungsi prediksi dengan rule tambahan
def prediction(processed_data, input_array):
    # --- RULE OVERRIDE ---
    tuition_fees = input_array[11]
    grade_avg = (float(input_array[4]) + float(input_array[5])) / 2
    approved_total = int(input_array[2]) + int(input_array[3])

    if tuition_fees == "Ya" and grade_avg >= 10 and approved_total >= 7:
        return "Graduate"

    # --- PREDIKSI MODEL ---
    pred = model.predict(processed_data)[0]
    return "Graduate" if pred == 0 else "Dropout"
