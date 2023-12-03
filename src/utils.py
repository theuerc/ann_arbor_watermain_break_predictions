import numpy as np 
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
        # convert to bool
        dummies = dummies * 1
        df = pd.concat([df, dummies], axis = 1)
    if drop:
        df = df.drop(columns = cols)
    return df


def break_timedelta(s, break_col: str, cutoff):
    '''
    apply rowwise to a pandas dataframe
    gets number of days between installation and the break 

    ARGS:
        s: pandas series (a row)
        break_col: column with information about breaks 
    '''
    if pd.isnull(s[break_col]):
        return (cutoff - s['INSTALLDAT']).days
    else:
        delta = (s[break_col] - s['INSTALLDAT']).days
        return delta
    


def process_date_cols(df: DataFrame, CUTOFF) -> DataFrame:
    '''
    handles datetimes, makes sure no illicit information 
    is in the training set 

    ARGS:
        df: dataframe with raw data
        CUTOFF: date cutoff for prediction 

    RETURNS: 
        new dataframe with dates formatted & additional columns
    '''
    df = df[pd.to_datetime(df['INSTALLDAT']) <= CUTOFF]
    df['first_break'] = pd.to_datetime(df['first_break'])
    df['most_recent_break'] = pd.to_datetime(df['most_recent_break'])
    df['INSTALLDAT'] = pd.to_datetime(df['INSTALLDAT'])

    df['all_breaks'] = df['all_breaks'].astype(str).apply(lambda s: s.split(","))
    df['all_breaks'] = df['all_breaks'].apply(lambda s: [pd.to_datetime(t) for t in s]) 

    df['breaks_before_cutoff'] = df['all_breaks'].apply(lambda s: [t for t in s if t <= CUTOFF])
    df['breaks_after_cutoff'] = df['all_breaks'].apply(lambda s: [t for t in s if t > CUTOFF])

    df['first_break'] = df['first_break'].apply(lambda s: np.where(s <= CUTOFF, s, pd.NaT))
    df['most_recent_break'] = df['most_recent_break'].apply(lambda s: np.where(s <= CUTOFF, s, pd.NaT))

    df['will_break'] = (df['breaks_after_cutoff'].apply(len) > 0).astype(int)

    return df



def svm_data_transform_pipeline(df: DataFrame, CUTOFF, dummy_cols: list[str]) -> DataFrame:
    '''
    fun little pipeline for svm dataset (could be used with other models too)

    ARGS:
        df = dataframe, has already had date columns cleaned / formatted
        CUTOFF: date cutoff for prediction 
        dummy_cols: list of columns to generate dummy variables for 

    RETURNS:
        dataframe that's ready for scaling & prediction 
    '''
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
    df['delta_installation_to_first_break'] = df.apply(lambda s: break_timedelta(s, 'first_break', CUTOFF), axis = 1)
    df['delta_installation_to_most_recent_break'] = df.apply(lambda s: break_timedelta(s, 'most_recent_break', CUTOFF), axis = 1)


    # drop nulls in categorical cols 
    df = df.dropna(subset = ['PressureSy', 'SUBTYPE'])

    # handle categorical columns
    df['SUBTYPE'] = df['SUBTYPE'].map({1: 'Distribution Main', 2: 'Transmission Main', 3: 'Hydrant Lead', 
                                       4: 'Raw Water', 5: 'Other', 6: 'Other'})
    
    df = convert_dummies(dummy_cols, df)

    to_drop = ['ENABLED', 'FACILITYID', 'LOCATION', 'INSTALLDAT', 'SUBTYPE', 'MATERIAL', 'STATUS', 'PressureSy', 
               'first_break', 'most_recent_break', 'all_breaks', 'break_status', 'breaks_before_cutoff', 
               'breaks_after_cutoff']
    df = df.drop(columns = to_drop)
    return df
