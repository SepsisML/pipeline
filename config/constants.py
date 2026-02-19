
# Core values
LAB_ATTRIBUTES = [
    "pH", "PaCO2", "AST", "BUN", "Alkalinephos", "Chloride", "Creatinine",
    "Lactate", "Magnesium", "Potassium", "Bilirubin_total", "PTT", "WBC",
    "Fibrinogen", "Platelets"
]


## No se agregó Etco2 porque no habían datos en el dataset
VITAL_ATTRIBUTES = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]

DEMOGRAPHIC_ATTRIBUTES = ["Age", "ICULOS", "Gender"]


FEATURES = (
    LAB_ATTRIBUTES
    + VITAL_ATTRIBUTES
    + DEMOGRAPHIC_ATTRIBUTES
)

## SIRS, qSOFA constants
SIRS_THRESHOLDS = {
    "temp_high": 38,
    "temp_low": 36,
    "hr": 90,
    "resp": 20,
    "wbc_high": 12000,
    "wbc_low": 4000,
}

QSOFA_THRESHOLDS = {
    "resp": 22,
    "sbp": 100,
}
