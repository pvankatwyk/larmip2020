from larmip import run_all_larmip, project_ismip, plot_larmip_results, project
import larmip
from generate_gsat_from_ismip import group_ismip_by_larmip_region
import pandas as pd
import numpy as np
from tqdm import tqdm
from ise.utils.functions import unscale

# results = run_all_larmip(num_iters=250)

all_dfs = []
num_iters = 10000

path = f'./train_with_gsat.csv'
data = pd.read_csv(path)
data = group_ismip_by_larmip_region(data, )


for id_ in tqdm(list(data.id.unique()), total=len(list(data.id.unique()))):
    model_run = data.loc[data.id == id_]
    model = model_run.larmip_model.values[0]
    region = model_run.larmip_region.values[0]
    GSAT = model_run.gsat
    sle = model_run.sle / 1000 * -1

    SL_wTd_nos_base = project_ismip(GSAT, model, num_iters=num_iters, region=region)
    results = pd.DataFrame(SL_wTd_nos_base.T, columns=[f'iter_{i+1}' for i in range(num_iters+1)])
    
    results['year'] = model_run.year.values
    results['model'] = model
    results['scenario'] = model_run.Scenario.values
    results['larmip_region'] = region
    if results.isna().any().any():
        stop = ''
    all_dfs.append(results)
    plot_larmip_results(SL_wTd_nos_base, model, scenario=model_run.Scenario.values[0].replace('_', "").upper(), true=sle.values, export_dir=f'./plots/')
    stop = ''
pd.concat(all_dfs).to_csv(f'./results_{num_iters}.csv', index=False)

path = f'./larmip_data.csv'
data = pd.read_csv(path)
data = group_ismip_by_larmip_region(data, )
data.rename(columns={'Scenario': 'scenario'}, inplace=True)


results_dataset = pd.concat(all_dfs)
proj_cols = [x for x in results_dataset.columns if 'iter' in x]
projs = results_dataset[proj_cols].values
results_dataset['mean'] = projs.mean(axis=1)
results_dataset['std'] = projs.std(axis=1)
results_dataset = results_dataset.drop(columns=proj_cols)
results_dataset.rename(columns={'model': 'larmip_model'}, inplace=True)

data = pd.merge(data, results_dataset, on=['year', 'larmip_model', 'scenario', 'larmip_region'], how='inner')
data.rename(columns={'mean': 'pred'}, inplace=True)
data['pred'] = data.pred * 1000 # convert to mm
data.to_csv(r'larmip_predictions.csv', index=False)
