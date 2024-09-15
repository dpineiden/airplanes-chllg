import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union, List

def read_data(
    path:Path, sep:str=',')->pd.DataFrame:
    """
    Read dataset from path 

    Args:
    path :: path to csv file
    sep :: separator

    Returns:
    pd.Dataframe
    Error if path can't be readed
    """
    try:
        return pd.read_csv(path, sep=sep)
    except Exception as e:
        print("Can't read data from csv",e)
        raise e
