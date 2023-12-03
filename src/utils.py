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


def break_timedelta(s, break_col: str):
    '''
    apply rowwise to a pandas dataframe
    gets number of days between installation and the break 

    ARGS:
        s: pandas series (a row)
        break_col: column with information about breaks 
    '''
    if pd.isnull(s[break_col]):
        return 0
    else:
        delta = (s[break_col] - s['INSTALLDAT']).days
        return delta
    


def svm_data_transform_pipeline(df: DataFrame, cols: list[str], drop:bool = True) -> DataFrame:
    # add date related features
    df['installation_year'] = df['INSTALLDAT'].dt.year
    df['n_previous_breaks'] = df['breaks_before_cutoff'].apply(len)

    # handle missing values in installdat
    # using median installation date for that material
    # using median instead of mean so we don't get fractions 
    df['installation_year'] = df['installation_year'].fillna(df.groupby("MATERIAL")['installation_year'].mean())
    # 24 instances where material & installation year are null, just dropping these 
    df = df.dropna(subset = ['installation_year', 'MATERIAL'])
    # finally converting to an int now that there's no na
    df['installation_year'] = df['installation_year'].astype(int)

    # time delta features
    df['delta_installation_to_first_break'] = df.apply(lambda s: break_timedelta(s, 'first_break'), axis = 1)
    df['delta_installation_to_most_recent_break'] = df.apply(lambda s: break_timedelta(s, 'most_recent_break'), axis = 1)


    # drop nulls in categorical cols 
    df = df.dropna(subset = ['PressureSy', 'SUBTYPE'])

    # handle categorical columns
    df['SUBTYPE'] = df['SUBTYPE'].map({1: 'Distribution Main', 2: 'Transmission Main', 3: 'Hydrant Lead', 
                                       4: 'Raw Water', 5: 'Other', 6: 'Other'})
    
    df = convert_dummies(cols, df, drop)
    return df
