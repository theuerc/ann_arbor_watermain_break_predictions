import pandas as pd
from pandas import DataFrame 


def convert_dummies(cols: list, df: DataFrame) -> DataFrame:
    for col in cols: 
        dummies = pd.get_dummies(df[col], drop_first = True)
        df = pd.concat([df, dummies], axis = 1)
    
    df = df.drop(columns = cols)
    return df