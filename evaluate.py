import pandas as pd
from generate_gsat_from_ismip import group_ismip_by_larmip_region
from ise.utils import functions as f
import matplotlib.pyplot as plt
from ise.evaluation import metrics as m
import numpy as np




path = f'./larmip_data.csv'
data = pd.read_csv(path)
data = group_ismip_by_larmip_region(data, )
data.rename(columns={'Scenario': 'scenario'}, inplace=True)


results_dataset = pd.read_csv('./results_1000.csv')
proj_cols = [x for x in results_dataset.columns if 'iter' in x]
projs = results_dataset[proj_cols].values
results_dataset['mean'] = projs.mean(axis=1)
results_dataset['std'] = projs.std(axis=1)
results_dataset = results_dataset.drop(columns=proj_cols)
results_dataset.rename(columns={'model': 'larmip_model'}, inplace=True)

data = pd.merge(data, results_dataset, on=['year', 'larmip_model', 'scenario', 'larmip_region'], how='inner')
data.rename(columns={'mean': 'pred'}, inplace=True)

# data['sle'] = f.unscale(data.sle.values.reshape(-1,1), './scaler_y.pkl') * 1000
# data['pred'] = f.unscale(data.pred.values.reshape(-1,1), './scaler_y.pkl') * 1000
data['pred'] = data.pred * 1000 # convert to mm



plt.figure(figsize=(10, 6))
plt.hist(data.sle, bins=30, alpha=0.5, label='True', density=True, color='blue')
plt.hist(data.pred, bins=20, alpha=0.5, label='LARMIP', density=True, color='red')

# Adding labels and title
plt.xlabel('LARMIP v ISMIP Projections')
plt.ylabel('Probability Density')
plt.title('Normalized Histogram of Sea Level Equivalent (SLE)')
plt.xlim([-10, 10])
plt.legend()
plt.savefig('./plots/larmip_vs_ismip_hist.png')


print(f"""MSE: {np.mean((data.pred - data.sle)**2)}
MAE: {np.mean(np.abs(data.pred - data.sle))}""")


ks = m.kolmogorov_smirnov
rcp26 = data.loc[data.scenario == 'rcp2.6']
rcp85 = data.loc[data.scenario == 'rcp8.5']

print(f"""KS Test (pred): {ks(rcp26.pred, rcp85.pred)}
KS Test (true): {ks(rcp26.sle, rcp85.sle)}""")

def cohen_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / pooled_std
  
cohen_d(rcp26.pred, rcp85.pred)

stop = ''