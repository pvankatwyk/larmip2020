
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from ise.utils.functions import get_all_filepaths
import difflib
import larmip

def match_file_paths(list1, list2):
    matched_paths = []
    for path in list1:
        # Get the closest match from list2 for the current path
        closest_match = difflib.get_close_matches(path, list2, n=1)
        if closest_match:  # Check if there is at least one match
            matched_paths.append((path, closest_match[0]))
        else:
            matched_paths.append((path, None))
    return matched_paths
  
def get_gsat_from_file(fp,):
    data = xr.open_dataset(fp)
    
    if 'anomaly' in fp:
        ts_anomaly_data = data.ts_anomaly.mean(dim=['lat', 'lon'])
        ts_anomaly_data = ts_anomaly_data[-86:]
        return ts_anomaly_data
    elif 'clim' in fp:
        ts_data = data.ts_clim.mean(dim=['lat', 'lon'])
        ts_data = ts_data.expand_dims(dim={'time': 86},)
        return ts_data
    else:
      raise ValueError("File path does not contain 'anomaly' or 'clim'")
  
  
def generate_gsat(atmospheric_forcing_dir=r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Atmosphere_Forcing/", with_clim=True):
  # get all fps to Original CMIP GSAT
  all_fps = get_all_filepaths(atmospheric_forcing_dir, contains=['Original_CMIP5_Grid', '1995-2100'])
  if with_clim:
    clim_fps = get_all_filepaths(atmospheric_forcing_dir, contains=['Original_CMIP5_Grid', '_clim_'])
    all_fps = match_file_paths(all_fps, clim_fps)
  
  df = []
  for fp in all_fps:
      if with_clim:
        anomaly_fp = fp[0] if 'anomaly' in fp[0] else fp[1]
        clim_fp = fp[1] if 'clim' in fp[1] else fp[0]
        total_gsat = get_gsat_from_file(clim_fp) + get_gsat_from_file(anomaly_fp) - 273.15 # to celsius
        df.append(pd.Series(total_gsat, name=anomaly_fp.split('/')[-1]),)

      else:
        total_gsat = get_gsat_from_file(fp)
        df.append(pd.Series(total_gsat, name=fp.split('/')[-1]),)
    
  df = pd.DataFrame(df,).T

  gsat_to_regional_mapping = {
    'NorESM1-M_anomaly_1995-2100.nc': 'noresm1-m_rcp85',
    'NorESM1-M_anomaly_rcp26_1995-2100.nc': 'noresm1-m_rcp26',
    'MIROC-ESM-CHEM_anomaly_1995-2100.nc': 'miroc-esm-chem_rcp85',
    'MIROC-ESM-CHEM_anomaly_rcp26_1995-2100.nc': 'miroc-esm-chem_rcp26',
    'CCSM4_anomaly_1995-2100.nc': 'ccsm4_rcp85',
    'CCSM4_anomaly_rcp26_1995-2100.nc': 'ccsm4_rcp26',
    'HadGEM2-ES_anomaly_rcp85_1995-2100.nc': 'hadgem2-es_rcp85',
    'HadGEM2-ES_anomaly_rcp26_1995-2100.nc': 'hadgem2-es_rcp26',
    'CSIRO-Mk3-6-0_anomaly_rcp85_1995-2100.nc': 'csiro-mk3.6_rcp85',
    'IPSL-CM5A-MR_anomaly_rcp85_1995-2100.nc': 'ipsl-cm5-mr_rcp85',
    'IPSL-CM5A-MR_anomaly_rcp26_1995-2100.nc': 'ipsl-cm5-mr_rcp26',
  }

  df.rename(columns=gsat_to_regional_mapping, inplace=True)

  regional_data = pd.read_csv(r'/oscar/scratch/pvankatw/datasets/sectors/AIS/with_chars/q0.5/train.csv')
  assert set(regional_data.aogcm.unique()) - set(df.columns) == set(), "Missing columns in the regional data"
  df['year'] = np.arange(2015, 2101)

  return df

def add_gsat_to_regional(regional_data, gsat):
  gsat['year'] = regional_data.year.unique()
  gsat_melted = gsat.melt(id_vars=['year'], var_name='aogcm', value_name='gsat')

  # Merging df2 with the melted df1 to add the GSAT values
  regional_data = pd.merge(regional_data, gsat_melted, on=['aogcm', 'year'], how='left')
  return regional_data

def group_ismip_by_larmip_region(data):
    
    # subset dataset to runs in LARMIP
    data = data[data.model.isin(larmip.ismip_models)]
    data['larmip_model'] = data.model.map(larmip.ismip_larmip_name_map)
    
    # convert normalized sectors to 1 to 18
    sector_to_id = {sector: id for id, sector in enumerate(data['sector'].unique(), start=1)}
    data['sector_original'] = data.sector.map(sector_to_id)
    
    # add LARMIP regions
    data['larmip_region'] = data.sector_original.map(larmip.ismip_sectors_to_larmip_regions)

    # subset to only columns of interest
    data = data[['larmip_model', 'year', 'gsat', 'Scenario', 'sector_original', 'larmip_region', 'id', 'sle']]
    
    # aggregate by LARMIP region, taking average over sector forcings and sum over sector SLE
    agg_funcs = {
        'gsat': 'mean',
        'sle': 'sum',
    }
    grouped = data.groupby(['larmip_model', 'larmip_region',  'Scenario', 'year']).agg(agg_funcs)
    grouped = grouped.reset_index()
    
    # create new ID
    grouped['id'] = grouped.larmip_model + '_' + grouped.larmip_region + '_' + grouped.Scenario
    
    return grouped
  
def process_ismip_to_larmip(ismip_dataset_fp, forcing_dir, export_dir=None):
  gsat = generate_gsat(forcing_dir)
  ismip = pd.read_csv(ismip_dataset_fp)
  joined = add_gsat_to_regional(ismip, gsat)
  
  if export_dir:
    joined.to_csv(f"{export_dir}/larmip_data.csv", index=False)
  return joined
  
  


if __name__ == "__main__":
  ismip_dataset_fp = r'/oscar/scratch/pvankatw/datasets/sectors/AIS/with_chars/q0.5/val.csv'
  forcing_dir = r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Atmosphere_Forcing/"
  process_ismip_to_larmip(ismip_dataset_fp, forcing_dir, export_dir='./')
  stop = ''