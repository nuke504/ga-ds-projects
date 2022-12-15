import pandas as pd
import numpy as np
import string

from typing import Dict, List, Tuple

BEER_CATEGORY_CUTOFF = [
    ("light",0.04),
    ("regular",0.06),
    ("strong",1)
]
'''
BEER_STYLE_MAP = {
    'abbey single ale':'ale',
    'altbier':'others',
    'american adjunct lager':'lager',
    'american amber  red ale':'ale',
    'american amber  red lager':'lager',
    'american barleywine':'strong ale',
    'american black ale':'ale',
    'american blonde ale':'ale',
    'american brown ale':'ale',
    'american dark wheat ale':'ale',
    'american double  imperial ipa':'ipa',
    'american double  imperial pilsner':'pilsner',
    'american double  imperial stout':'stout',
    'american india pale lager':'lager',
    'american ipa':'ipa',
    'american malt liquor':'others',
    'american pale ale apa':'ale',
    'american pale lager':'lager',
    'american pale wheat ale':'ale',
    'american pilsner':'pilsner',
    'american porter':'porter',
    'american stout':'stout',
    'american strong ale':'strong ale',
    'american white ipa':'ipa',
    'american wild ale':'ale',
    'baltic porter':'porter',
    'belgian dark ale':'ale',
    'belgian ipa':'ipa',
    'belgian pale ale':'ale',
    'belgian strong dark ale':'strong ale',
    'belgian strong pale ale':'strong ale',
    'berliner weissbier':'weissbier',
    'bière de garde':'others',
    'bock':'others',
    'braggot':'others',
    'california common  steam beer':'others',
    'chile beer',
    'cider',
    'cream ale',
    'czech pilsener',
    'doppelbock',
    'dortmunder  export lager',
    'dubbel',
    'dunkelweizen',
    'english barleywine',
    'english bitter',
    'english brown ale',
    'english dark mild ale',
    'english india pale ale ipa',
    'english pale ale',
    'english pale mild ale',
    'english stout',
    'english strong ale',
    'euro dark lager',
    'euro pale lager',
    'extra special  strong bitter esb',
    'flanders oud bruin',
    'flanders red ale',
    'foreign  export stout',
    'fruit  vegetable beer',
    'german pilsener',
    'gose',
    'grisette',
    'hefeweizen',
    'herbed  spiced beer',
    'irish dry stout',
    'irish red ale',
    'keller bier  zwickel bier',
    'kristalweizen',
    'kölsch',
    'light lager',
    'low alcohol beer',
    'maibock  helles bock',
    'mead',
    'milk  sweet stout',
    'munich dunkel lager',
    'munich helles lager',
    'märzen  oktoberfest',
    'oatmeal stout',
    'old ale',
    'other',
    'pumpkin ale',
    'quadrupel quad',
    'radler',
    'rauchbier',
    'roggenbier',
    'russian imperial stout',
    'rye beer',
    'saison  farmhouse ale',
    'schwarzbier',
    'scotch ale  wee heavy',
    'scottish ale',
    'shandy',
    'smoked beer',
    'tripel',
    'vienna lager',
    'wheat ale',
    'winter warmer',
    'witbier'
}
'''
def add_beer_category(
    df:pd.DataFrame,
    abv_col:str = "abv",
    beer_categories: List[Tuple[str, float]] = BEER_CATEGORY_CUTOFF
) -> pd.DataFrame:
    """
    Adds the beer category column
    """

    def helper(abv):
        if pd.isnull(abv):
            return np.NaN

        for cat, cutoff in beer_categories:
            if abv <= cutoff:
                return cat

        return np.NaN

    df["beer_category"] = df[abv_col].apply(helper)

    assert df["beer_category"].isna().sum() == df[abv_col].isna().sum(), "There are null values in the category"

    return df



def read_dataframes(
    df_beer_addr: str,
    df_brewery_addr: str,
    df_state_addr: str
) -> Dict[str, pd.DataFrame]:
    """
    Read dataframes from links and preprocesses them
    """

    # Read all datasets and preprocess them
    df_beer = pd.read_csv(df_beer_addr).iloc[:,1:]
    df_brewery = pd.read_csv(df_brewery_addr)
    df_state = pd.read_csv(df_state_addr).rename(columns = {"State Code":"brewery_state","State":"state_name","Region":"region","Division":"division"})

    df_brewery.columns = ["brewery_id", *[f"brewery_{colname}" for colname in df_brewery.columns[1:]]]
    df_brewery["brewery_state"] = df_brewery.brewery_state.str.strip()

    return {
        "beer": df_beer,
        "brewery": df_brewery,
        "state": df_state
    }


def merge_data(
    df_beer: pd.DataFrame,
    df_brewery: pd.DataFrame,
    df_state: pd.DataFrame
) -> pd.DataFrame:
    """
    Preprocess the dataset

    Perform the following procedures

    1. Join the dataframes

    Join brewery with state and beer shape


    """
    largest_num_rows = max(df_beer.shape[0], df_brewery.shape[0], df_state.shape[0])

    # Join the two datasets
    df_bb = df_beer.merge(df_brewery, how="left", on="brewery_id").merge(df_state, how="left", on="brewery_state")
    
    assert df_bb.shape[0] == largest_num_rows, "Result number of rows not equal to the largest dataframe"

    return df_bb


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset

    Process

    For the following string columns:
    - name: lowercase + remove punctuation
    - style: lowercase + remove punctuation
    - brewery_name: lowercase + remove punctuation
    - brewery_city: lowercase + remove punctuation
    """

    df = df.copy()

    # Drop NA 
    df = df[~df["style"].isna()].copy()

    def process_text_col(col: pd.Series) -> pd.Series:
        """
        Process text column by lowercase + remove punctuation
        """
        col = col.str.lower()
        col = col.apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))
        return col


    for col in ["name","style","brewery_name","brewery_city"]:
        df[col] = process_text_col(df[col])

    return df


