from larmip import run_all_larmip, project_ismip, plot_larmip_results, project
import larmip
import pandas as pd
import numpy as np
from tqdm import tqdm
from ise.utils.functions import unscale

# results = run_all_larmip(num_iters=250)

all_dfs = []
num_iters = 200
with_clim = True
path = f'./withclim_{with_clim}_train_with_gsat.csv'
data = pd.read_csv(path)
data = data[data.model.isin(larmip.ismip_models)]
data['larmip_model'] = data.model.map(larmip.ismip_larmip_name_map)
sector_to_id = {sector: id for id, sector in enumerate(data['sector'].unique(), start=1)}


data['sector_original'] = data.sector.map(sector_to_id)
data['larmip_region'] = data.sector_original.map(larmip.ismip_sectors_to_larmip_regions)
# data['sle'] = unscale(data.sle.values.reshape(-1,1), './scaler_y.pkl')

data = data[['larmip_model', 'year', 'gsat', 'Scenario', 'sector_original', 'larmip_region', 'id', 'sle']]
agg_funcs = {
    'gsat': 'mean',
    'sle': 'sum',
}

grouped = data.groupby(['larmip_model', 'larmip_region',  'Scenario', 'year']).agg(agg_funcs)
grouped = grouped.reset_index()
grouped['id'] = grouped.larmip_model + '_' + grouped.larmip_region + '_' + grouped.Scenario
data = grouped
# model = np.random.choice(list(process.ismip_larmip_name_map.values()))
# model_runs = data.loc[data.larmip_model == model].reset_index(drop=True)
for id_ in tqdm(list(data.id.unique()), total=len(list(data.id.unique()))):
    model_run = data.loc[data.id == id_]
    model = model_run.larmip_model.values[0]
    region = model_run.larmip_region.values[0]
    GSAT = model_run.gsat
    sle = model_run.sle / 1000 * -1

    SL_wTd_nos_base = project_ismip(GSAT, model, num_iters=num_iters, region=region)
    results = pd.DataFrame(SL_wTd_nos_base.T, columns=[f'iter_{i+1}' for i in range(num_iters+1)])
    results['year'] = model_run.year
    results['model'] = model
    results['scenario'] = model_run.Scenario
    all_dfs.append(results)
    plot_larmip_results(SL_wTd_nos_base, model, scenario=model_run.Scenario.values[0].replace('_', "").upper(), true=sle.values )
    stop = ''
pd.concat(all_dfs).to_csv(f'./results_{num_iters}.csv', index=False)
    
# test_case = model_runs.loc[model_runs.id == np.random.choice(model_runs.id.unique())]
# GSAT = test_case.gsat
# sle = test_case.sle / 1000

# train = pd.read_csv(r'./train_with_gsat.csv')
# SL_wTd_nos_base_SU = project_ismip(GSAT, model, num_iters=100)
# all_dfs.append(pd.DataFrame(SL_wTd_nos_base_SU))
# plot_larmip_results(SL_wTd_nos_base_SU, model, scenario='RCP85', true=sle.values )

# # Why did they use MAGICC instead of the actual CMIP values?
# # All of the values are basically zero when converted to meters. Is something going wrong?
# stop = ''