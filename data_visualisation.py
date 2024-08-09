from statistics import MetricCalc
from models import *
import numpy as np
from tools import *
import plotly.graph_objs as go
from matplotlib import pyplot as plt


''' visualisation of parameters '''

def visual_mean(df:pd.DataFrame, mc: MetricCalc, name:str):
    mean20 = []
    for i in range(df.shape[0] - 20):
        mean20.append(mc.mean(df.loc[i:i+20]))
    mean50 = []
    for i in range(df.shape[0] - 50):
        mean50.append(mc.mean(df.loc[i:i+50]))
    mean100 = []
    for i in range(df.shape[0] - 100):
        mean100.append(mc.mean(df.loc[i:i+100]))
    mean150 = []
    for i in range(df.shape[0] - 150):
        mean150.append(mc.mean(df.loc[i:i+150]))

    fig = go.Figure()
    fig.update_layout(title="MEAN OF " + name + ' ' + mc.col)

    fig.update_xaxes(title='date')
    fig.update_yaxes(title='mean index value')

    fig.add_trace(go.Scatter(x=[df.at[i, "Дата"] for i in range(len(mean20))], y=mean20, name="x20"))
    fig.add_trace(go.Scatter(x=[df.at[i, "Дата"] for i in range(len(mean50))], y=mean50, name="x50"))
    fig.add_trace(go.Scatter(x=[df.at[i, "Дата"] for i in range(len(mean100))], y=mean100, name="x100"))
    fig.add_trace(go.Scatter(x=[df.at[i, "Дата"] for i in range(len(mean150))], y=mean150, name="x150"))

    fig.show()

def visual_asymmetry(df:pd.DataFrame, mc: MetricCalc, name:str):
    # asymmetry20 = []
    # for i in range(df.shape[0] - 20):
    #     asymmetry20.append(mc.asymmetry(df.loc[i:i+20]))
    asymmetry50 = []
    for i in range(df.shape[0] - 50):
        asymmetry50.append(mc.asymmetry(df.loc[i:i+50]))
    asymmetry100 = []
    for i in range(df.shape[0] - 100):
        asymmetry100.append(mc.asymmetry(df.loc[i:i+100]))
    asymmetry150 = []
    for i in range(df.shape[0] - 150):
        asymmetry150.append(mc.asymmetry(df.loc[i:i+150]))

    fig = go.Figure()
    fig.update_layout(title="ASYMMETRY OF " + name + ' ' + mc.col)

    fig.update_xaxes(title='date')
    fig.update_yaxes(title='asymmetry coeff')


    fig.add_trace(go.Scatter(x=[df.at[i, "Дата"] for i in range(len(asymmetry50))], y=asymmetry50, name="x50"))
    fig.add_trace(go.Scatter(x=[df.at[i, "Дата"] for i in range(len(asymmetry100))], y=asymmetry100, name="x100"))
    fig.add_trace(go.Scatter(x=[df.at[i, "Дата"] for i in range(len(asymmetry150))], y=asymmetry150, name="x150"))

    fig.show()

def visual_excess(df:pd.DataFrame, mc: MetricCalc, name:str):
    # excess20 = []
    # for i in range(df.shape[0] - 20):
    #     excess20.append(mc.excess(df.loc[i:i+20]))
    excess50 = []
    for i in range(df.shape[0] - 50):
        excess50.append(mc.excess(df.loc[i:i+50]))
    excess100 = []
    for i in range(df.shape[0] - 100):
        excess100.append(mc.excess(df.loc[i:i+100]))
    excess150 = []
    for i in range(df.shape[0] - 150):
        excess150.append(mc.excess(df.loc[i:i+150]))

    fig = go.Figure()
    fig.update_layout(title="EXCESS OF " + name + ' ' + mc.col)

    fig.update_xaxes(title='date')
    fig.update_yaxes(title='excess coeff')


    fig.add_trace(go.Scatter(x=[df.at[i, "Дата"] for i in range(len(excess50))], y=excess50, name="x50"))
    fig.add_trace(go.Scatter(x=[df.at[i, "Дата"] for i in range(len(excess100))], y=excess100, name="x100"))
    fig.add_trace(go.Scatter(x=[df.at[i, "Дата"] for i in range(len(excess150))], y=excess150, name="x150"))

    fig.show()

def visual_arma(df: pd.DataFrame, name: str, model):
    if not 'Residual' in df:
        predicted_values = model.predict(start=df['Прибыль'].index[0], end=df['Прибыль'].index[-1], typ='levels')
        df['Residual'] = df['Прибыль'] - predicted_values
    plt.figure(figsize=(10, 6))
    plt.title("True vs Predicted Values for SSE")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.plot(df['Дата'], df['Прибыль'], label="True Values", color="blue")
    plt.plot(df['Дата'], df['Прибыль'] - df['Residual'], label="Predicted Values", color="red")
    plt.xticks(np.arange(min(df.index), max(df.index), df.shape[0] // 8))
    plt.legend()
    plt.show()

def visual_garch(df: pd.DataFrame, name: str, model):
    if not 'Volatility' in df:
        df['Volatility'] = model.conditional_volatility
    if not 'AnnualVol' in df:
        df['AnnualVol'] = df['Volatility'] * np.sqrt(250)
    plt.figure(figsize=(10, 6))
    plt.title("Residuals and Volatility Values for " + name)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.plot(df['Дата'], abs(df['Residual']), label="Residuals", color='blue')
    plt.plot(df['Дата'], model.conditional_volatility, label="Volatility", color='red')
    plt.xticks(np.arange(min(df.index), max(df.index), df.shape[0] // 8))
    plt.legend()
    plt.show()

def visual_tgarch(df: pd.DataFrame, name: str, model):
    if not 'TVolatility' in df:
        df['TVolatility'] = model.conditional_volatility
    if not 'AnnualTVol' in df:
        df['AnnualTVol'] = df['TVolatility'] * np.sqrt(250)
    plt.figure(figsize=(10, 6))
    plt.title("Residuals and Volatility Values for " + name)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.plot(df['Дата'], abs(df['Residual']), label="Residuals", color='blue')
    plt.plot(df['Дата'], model.conditional_volatility, label="TVolatility", color='red')
    plt.xticks(np.arange(min(df.index), max(df.index), df.shape[0] // 8))
    plt.legend()
    plt.show()
