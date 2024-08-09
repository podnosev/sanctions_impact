import pandas as pd
import numpy as np
import chardet
import re

def str_to_float(str):
    return float(str.replace('.','').replace(',', '.'))

def bad_CSV(PATH:str, NEW_PATH:str):
    with open(PATH, 'rb') as f:
        result = chardet.detect(f.read())
    set = pd.read_csv(PATH, encoding=result['encoding'])
    print(set)
    for col in set.columns:
        set = set[col].str.split(',"', expand=True)
    print(set)
    set.columns=['Дата', 'Цена', 'Откр.', 'Макс.', 'Мин.', 'Объём', 'Изм. %']
    print(set)
    for colum in set.columns:
        set[colum] = set[colum].apply(lambda x: re.sub("\s+", "", x))
        set[colum] = set[colum].apply(lambda x: re.sub('"', "", x))
    print(set)
    set.to_csv(NEW_PATH, index=False, encoding='utf-8')

def read_CSV_data(PATH: str):
    try:
        set = pd.read_csv(PATH)
    except(UnicodeDecodeError):
        NEW_PATH = PATH.replace('.csv', '(utf-8).csv')
        bad_CSV(PATH, NEW_PATH)
        set = pd.read_csv(NEW_PATH)
    set = set.drop(columns=['Откр.', 'Макс.', 'Мин.', 'Объём', 'Изм. %'])
    set['Цена'] = set['Цена'].apply(str_to_float)
    set = set.iloc[::-1].reset_index(drop=True)
    set['Прибыль'] = np.log(set['Цена']).diff()*100
    set.at[0, 'Прибыль'] = 0
    return set