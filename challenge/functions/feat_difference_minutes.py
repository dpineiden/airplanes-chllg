import pandas as pd
from datetime import datetime


def get_min_diff(data:pd.DataFrame)->float:
    """Returns the minimum difference between dates between Fecha-0
    and Fecha-1.

    Args:
    data: dataframe with dataset

    Returns:
    min_diff: float, minutes difference    

    """
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds())/60
    return min_diff
