import pandas as pd
import numpy as np
from config import LAB_ATTRIBUTES, VITAL_ATTRIBUTES, DEMOGRAPHIC_ATTRIBUTES

def z_score_normalize(df: pd.DataFrame) -> pd.DataFrame:
    columns = (
        LAB_ATTRIBUTES
        + VITAL_ATTRIBUTES
        + [a for a in DEMOGRAPHIC_ATTRIBUTES if a != "Gender"] ## variable discreta
    )
    df_norm = df.copy()

    for col in columns:
        mean_val = df[col].mean(skipna=True)
        std_val = df[col].std(skipna=True)

        if std_val == 0:
            df_norm[col] = 0
        else:
            df_norm[col] = (df[col] - mean_val) / std_val

    return df_norm
