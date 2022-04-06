def obj_to_cat(data):
    '''Быстро преоброзовать данные из 
    типа данных "object" в тип данных
    "category"'''
    
    obj_col_cond = (data.dtypes == 'object')
    data.loc[:, obj_col_cond] =\
    data.loc[:, obj_col_cond].astype("category")
    
    return data
    
    
    