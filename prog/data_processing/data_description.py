import pandas as pd
import numpy as np

types_natural_names = {
    "datetime64[ns]" : 'Дата',
    "category" : "Номинативная",
    "bool" : "Номинативная",
    "int64" : "Целое число",
    "int32" : "Целое число",
    "float64": "Дейсвительное число"
}

def get_col_av_values(col):
    '''Получить область допустимых значений колонки'''
    if pd.api.types.is_numeric_dtype(col):
        return "[" + str(np.min(col)) + ";" + str(np.max(col)) + ']'
    if pd.api.types.is_datetime64_any_dtype(col):
        return "[" + np.min(col).strftime("%d.%m.%Y") + ";" +\
                np.max(col).strftime("%d.%m.%Y") + ']'
    else:
        return str(col.cat.categories.tolist())[1:-1].replace("'", "")

def get_col_cat_count(col):
    '''получить для категориальной переменной число
    уровней, а для численной "-"'''
    if pd.api.types.is_numeric_dtype(col) or\
        pd.api.types.is_datetime64_any_dtype(col):
        return "-"
    else:
        return len(col.cat.categories)

    
def get_data_frame_settings(df):
    
    result = pd.DataFrame(columns = [
        'Тип данных', 
        'Область допустимых значений',
        'Число допустимых значений',
        'Число пропусков'
    ])
    
    for col in df.columns:
        result.loc[col, :] = [
            df[col].dtype,
            get_col_av_values(df[col]),
            get_col_cat_count(df[col]),
            sum(df[col].isna())
        ]
    
    result["Тип данных"] = result["Тип данных"].replace(types_natural_names)
    return result