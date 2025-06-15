import numpy as np
import joblib
import pandas as pd

# Load model dan scaler sekali saja
model = joblib.load("model_dropout.pkl")
scaler = joblib.load("scaler.pkl")

# Mapping input kategori ke angka
def map_input(gender, debtor, tuition_fees, scholarship_holder):
    return [
        1 if gender == "Perempuan" else 0,
        1 if debtor == "Ya" else 0,
        1 if tuition_fees == "Ya" else 0,
        1 if scholarship_holder == "Ya" else 0
    ]

# Fungsi preprocessing input user
def data_preprocessing(input_array):
    numeric_part = np.array(input_array[:9], dtype=float)
    categorical_part = map_input(*input_array[9:])
    full_data = np.concatenate([numeric_part, categorical_part]).reshape(1, -1)

    # Skalakan seluruh fitur
    X_scaled = scaler.transform(full_data)
    df_scaled = pd.DataFrame(X_scaled, columns=[
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
    ])
    return df_scaled

# Fungsi prediksi
def prediction(processed_data):
    pred = model.predict(processed_data)[0]
    if pred == 1:
        return "Dropout"
    else:
        return "Graduate"