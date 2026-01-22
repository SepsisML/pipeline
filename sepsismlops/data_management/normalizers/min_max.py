import pandas as pd
import numpy as np
from config import LAB_ATTRIBUTES, VITAL_ATTRIBUTES, DEMOGRAPHIC_ATTRIBUTES

def min_max_normalize(df: pd.DataFrame) -> pd.DataFrame:
    columns = (
        LAB_ATTRIBUTES
        + VITAL_ATTRIBUTES
        + [a for a in DEMOGRAPHIC_ATTRIBUTES if a != "Gender"] ## variable discreta
    )
    
    df_norm = df.copy()

    for col in columns:
        min_val = df[col].min(skipna=True)
        max_val = df[col].max(skipna=True)

        if max_val == min_val:
            df_norm[col] = 0
        else:
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)

    return df_norm
