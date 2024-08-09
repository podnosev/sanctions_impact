import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import statsmodels
from pmdarima.arima import auto_arima
import pandas as pd
from arch import arch_model

def auto_arma(df: pd.DataFrame, col: str):
    model = auto_arima(df[col], start_p=2, start_q=2,
                      max_p=5, max_q=5, m=1, seasonal=False,
                      d=None, trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True,
                      n_jobs=-1)
    return model.order[0], model.order[2], model.arima_res_

def best_ARMA(df: pd.DataFrame, col: str):
    best_params = [0, 1, ARIMA(df[col], order=(0, 0, 1)).fit()]
    for a in range(2, 5):
        for b in range(2, 5):
            model = ARIMA(df[col], order=(a, 0, b))
            model_fit = model.fit()
            if best_params[2].aic > model_fit.aic:
                best_params = [a, b, model_fit]
    return best_params

def ARMA_df_update(df: pd.DataFrame):
    best_params = best_ARMA(df, 'Прибыль')
    predicted_values = best_params[2].predict(start=df['Прибыль'].index[0], end=df['Прибыль'].index[-1], typ='levels')
    df['Residual'] = df['Прибыль'] - predicted_values
    return best_params

def best_GARCH(df: pd.DataFrame, col: str):
    best_params = [1, 0, arch_model(df[col], vol='GARCH', p=1, q=0).fit()]
    for p in range(1, 5):
        for q in range(1, 5):
            model = arch_model(df[col], vol='GARCH', p=p, q=q)
            model_fit = model.fit()
            if best_params[2].aic > model_fit.aic:
                best_params = [p, q, model_fit]
    return best_params

def GARCH_df_update(df: pd.DataFrame):
    best_params = best_GARCH(df, 'Residual')
    df['Volatility'] = best_params[2].conditional_volatility
    df['AnnualVol'] = best_params[2].conditional_volatility * np.sqrt(250)
    df['GResidual'] = abs(df['Residual']) - df['Volatility']
    return best_params

def best_TGARCH(df: pd.DataFrame, col: str):
    best_params = [1, 0, arch_model(df[col], vol='GARCH', p=1, o=1, q=0, power=1.0).fit()]
    for p in range(1, 5):
        for q in range(1, 5):
            model = arch_model(df[col], vol='GARCH', p=p, o=p, q=q, power=1.0)
            model_fit = model.fit()
            if best_params[2].aic > model_fit.aic:
                best_params = [p, q, model_fit]
    return best_params

def TGARCH_df_update(df: pd.DataFrame):
    best_params = best_TGARCH(df, 'Residual')
    df['TVolatility'] = best_params[2].conditional_volatility
    df['AnnualTVol'] = best_params[2].conditional_volatility * np.sqrt(250)
    df['TResidual'] = abs(df['Residual']) - df['TVolatility']
    return best_params
