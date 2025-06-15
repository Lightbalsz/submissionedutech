import numpy as np
import joblib
import pandas as pd

# Load model dan scaler
model = joblib.load("model_dropout.pkl")
scaler = joblib.load("scaler.pkl")

# Mapping nilai kategorikal
def map_input(gender, debtor, tuition_fees, scholarship_holder):
    return [
        1 if gender == "Perempuan" else 0,
        1 if debtor == "Ya" else 0,
        1 if tuition_fees == "Ya" else 0,
        1 if scholarship_holder == "Ya" else 0
    ]

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

def data_preprocessing(input_array):
    # Pisahkan numerik & kategorikal
    numeric_part = np.array(input_array[:9], dtype=float)  # 9 numerik
    categorical_part = np.array(map_input(*input_array[9:]), dtype=int)  # 4 kategorikal
    full_input = np.concatenate([numeric_part, categorical_part]).reshape(1, -1)

    # Scaling
    scaled = scaler.transform(full_input)

    # Buat DataFrame dengan urutan kolom model
    df_scaled = pd.DataFrame(scaled, columns=FEATURE_ORDER)
    return df_scaled

def prediction(processed_data):
    pred = model.predict(processed_data)[0]
    return "Graduate" if pred == 0 else "Dropout"
