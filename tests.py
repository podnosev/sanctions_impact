from statistics import *
from data_visualisation import *
from matplotlib import pyplot as plt
from models import *

# print("mean = ", mc.mean(set))
# print("median = ", mc.median(set))
# print("max = ", mc.max(set))
# print("min = ", mc.min(set))
# print("std = ", mc.std(set))
# print("dispersion = ", mc.dispersion(set))
# print("asymmetry = ", mc.asymmetry(set))
# print("excess = ", mc.excess(set))
# print("drawdown from ", set.at[set.shape[0] // 2, "Дата"]," = ", mc.drawdown(set, set.shape[0] // 2))
# print("normal test stats = ", mc.jarque_bera(set))

set = read_CSV_data('C:/Users/main/Downloads/Shanghai Composite.csv')
print(set.head)
print(set.describe())


# mc = MetricCalc("Прибыль")
# # mc.col = "Доходность"
#
# name = 'SSE'
#
# visual_mean(set, mc, name)
# visual_asymmetry(set, mc, name)
# visual_excess(set, mc, name)

# # gets best ARMA model
model = ARMA_df_update(set)[2]
#model = ARIMA(set['Прибыль'], order=(2, 0, 4)).fit()
sp_col = set['Прибыль']
predicted_values = model.predict(start=sp_col.index[0], end=sp_col.index[-1], typ='levels')
set['Residual'] = set['Прибыль'] - predicted_values
print(set)
visual_arma(set, 'SSE', model)

# # gets best GARCH model
gmodel = GARCH_df_update(set)[2]
#gmodel = arch_model(set['Residual'], vol='GARCH', p=1, q=2).fit()
print(gmodel.summary())
visual_garch(set, 'SSE', gmodel)
print(set)

tmodel = TGARCH_df_update(set)[2]
print(tmodel.summary())
print(set)
visual_tgarch(set, 'SSE', tmodel)
