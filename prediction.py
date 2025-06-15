import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------------------------------------------------
# 1.  Muat artefak yang sudah diserialisasi (model & scaler)
# -----------------------------------------------------------------------
MODEL_PATH = Path(__file__).with_name("model_dropout.pkl")
SCALER_PATH = Path(__file__).with_name("scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------------------------------------------------
# 2.  Helper untuk mengonversi input kategori ke numerik (0/1)
# -----------------------------------------------------------------------
def map_input(gender, debtor, tuition_fees, scholarship_holder):
    """Konversi pilihan UI menjadi nilai biner."""
    return [
        1 if gender == "Perempuan" else 0,
        1 if debtor == "Ya" else 0,
        1 if tuition_fees == "Ya" else 0,
        1 if scholarship_holder == "Ya" else 0,
    ]

# -----------------------------------------------------------------------
# 3.  Fungsi pra-proses untuk data masukan dari Streamlit
# -----------------------------------------------------------------------
def data_preprocessing(raw_list):
    """
    Parameters
    ----------
    raw_list : list
        13 elemen dengan urutan:
        [curr_1st_enrolled, curr_2nd_enrolled, curr_1st_approved,
         curr_2nd_approved, curr_1st_grade, curr_2nd_grade,
         curr_1st_eval, curr_2nd_eval, age_at_enrollment,
         gender, debtor, tuition_fees, scholarship_holder]

    Returns
    -------
    pd.DataFrame
        Satu baris DataFrame yang sudah siap masuk ke model.predict
    """
    # --- Pisahkan numerik & kategorikal ---
    numeric_vals = np.asarray(raw_list[:9], dtype=float).reshape(1, -1)
    categorical_vals = np.asarray(
        map_input(*raw_list[9:]), dtype=int
    ).reshape(1, -1)

    # --- Scale hanya bagian numerik ---
    numeric_scaled = scaler.transform(numeric_vals)

    # --- Gabungkan kembali ---
    all_features = np.hstack([numeric_scaled, categorical_vals])

    # --- Buat DataFrame dengan nama kolom konsisten ---
    columns = [
        "Curricular_units_1st_sem_enrolled",
        "Curricular_units_2nd_sem_enrolled",
        "Curricular_units_1st_sem_approved",
        "Curricular_units_2nd_sem_approved",
        "Curricular_units_1st_sem_grade",
        "Curricular_units_2nd_sem_grade",
        "Curricular_units_1st_sem_evaluations",
        "Curricular_units_2nd_sem_evaluations",
        "Age_at_enrollment",
        "Gender",
        "Debtor",
        "Tuition_fees_up_to_date",
        "Scholarship_holder",
    ]
    return pd.DataFrame(all_features, columns=columns)

# -----------------------------------------------------------------------
# 4.  Fungsi prediksi akhir
# -----------------------------------------------------------------------
def prediction(processed_df):
    """
    Parameters
    ----------
    processed_df : pd.DataFrame
        Output dari data_preprocessing()

    Returns
    -------
    str
        'Dropout' jika model memprediksi 1, else 'Graduate'.
    """
    pred = model.predict(processed_df)[0]
    return "Dropout" if pred == 1 else "Graduate"