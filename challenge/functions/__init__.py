from .read_df import read_data
from .feat_get_period_day import get_period_day
from .feat_high_season import is_high_season
from .feat_difference_minutes import get_min_diff
import pandas as pd

import numpy as np
from pathlib import Path

def add_delay(
        data:pd.DataFrame,
        threshold_in_minutes:int=15)->None:
    """ 
    Void fn that add delay column
    Args:
    - df : dataframe with dataset
    - threshold_in_minutes: amount of minutes to delay

    Returns:
    - None
    """
    threshold_in_minutes = 15
    data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)


def process_dataframe(data:pd.DataFrame)->pd.DataFrame:
    """ 
    Given path, build dataframe based on path, adding extra fatures
    """
    try:
        data['period_day'] = data['Fecha-I'].apply(get_period_day)
        data['high_season'] = data['Fecha-I'].apply(is_high_season)
        data['min_diff'] = data.apply(get_min_diff, axis = 1)        
        add_delay(data)
    except Exception as ex:
        raise ex
