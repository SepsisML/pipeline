LAB_ATTRIBUTES = [
    "pH", "PaCO2", "AST", "BUN", "Alkalinephos", "Chloride", "Creatinine",
    "Lactate", "Magnesium", "Potassium", "Bilirubin_total", "PTT", "WBC",
    "Fibrinogen", "Platelets"
]

VITAL_ATTRIBUTES = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]

DEMOGRAPHIC_ATTRIBUTES = ["Age", "ICULOS", "Gender"]


FEATURES = (
    LAB_ATTRIBUTES
    + VITAL_ATTRIBUTES
    + DEMOGRAPHIC_ATTRIBUTES
)