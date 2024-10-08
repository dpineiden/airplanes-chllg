import pandas as pd

def get_rate_from_column(
        data:pd.DataFrame, 
        column:str)->pd.DataFrame:

    """
    Get rate from column

    Args:
    - data: dataframe
    - column: string

    Returns:
    - dataframe with rates
    """
    delays = {}
    for _, row in data.iterrows():
        if row['delay'] == 1:
            if row[column] not in delays:
                delays[row[column]] = 1
            else:
                delays[row[column]] += 1
    total = data[column].value_counts().to_dict()
    
    rates = {}
    for name, total in total.items():
        if name in delays:
            rates[name] = round(total / delays[name], 2)
        else:
            rates[name] = 0
            
    return pd.DataFrame.from_dict(data = rates, orient = 'index', columns = ['Tasa (%)'])
