import pandas as pd
from pandas import DataFrame 


def convert_dummies(cols: list, df: DataFrame, drop:bool = False) -> DataFrame:
    '''
    helper function to add dummy variables 

    ARGS:
        cols: list of columns to make dummies for 
        df: dataframe that contains 'cols'
        drop: boolean value for wether or not to drop the original columns
    
    RETURNS:
        new dataframe with dummy variables
    '''
    for col in cols: 
        dummies = pd.get_dummies(df[col], drop_first = True)
        df = pd.concat([df, dummies], axis = 1)
    
    if drop:
        df = df.drop(columns = cols)
    return df