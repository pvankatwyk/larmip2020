import pandas as pd
import numpy as np
from ise.evaluation import metrics as m

magic_data_dir = r"./larmip/MAGICC/RCP2500/"
rcp26 = pd.read_csv(magic_data_dir + "Temp_RCP26_HistConstraint.dat", sep='\s+', index_col=0, header=None).values.flatten()
rcp45 = pd.read_csv(magic_data_dir + "Temp_RCP45_HistConstraint.dat", sep='\s+', index_col=0, header=None).values.flatten()
rcp60 = pd.read_csv(magic_data_dir + "Temp_RCP60_HistConstraint.dat", sep='\s+', index_col=0, header=None).values.flatten()
rcp85 = pd.read_csv(magic_data_dir + "Temp_RCP85_HistConstraint.dat", sep='\s+', index_col=0, header=None).values.flatten()

ks = m.kolmogorov_smirnov
d_result = np.array(
  [
  [ 0,  ks(rcp26, rcp45)[0],  ks(rcp26, rcp60)[0], ks(rcp26, rcp85)[0], ],
  [ ks(rcp45, rcp26)[0], 0,  ks(rcp45, rcp60)[0], ks(rcp45, rcp85)[0], ],
  [ ks(rcp60, rcp26)[0],  ks(rcp60, rcp45)[0], 0,  ks(rcp60, rcp85)[0], ],
  [ ks(rcp85, rcp26)[0],  ks(rcp85, rcp45)[0],  ks(rcp85, rcp60)[0], 0, ],
  ]
  )
d_result = pd.DataFrame(d_result, columns=['rcp26', 'rcp45', 'rcp60', 'rcp85'], index=['rcp26', 'rcp45', 'rcp60', 'rcp85'])
p_result = np.array(
  [
  [ 0,  ks(rcp26, rcp45)[1],  ks(rcp26, rcp60)[1], ks(rcp26, rcp85)[1], ],
  [ ks(rcp45, rcp26)[1], 0,  ks(rcp45, rcp60)[1], ks(rcp45, rcp85)[1], ],
  [ ks(rcp60, rcp26)[1],  ks(rcp60, rcp45)[1], 0,  ks(rcp60, rcp85)[1], ],
  [ ks(rcp85, rcp26)[1],  ks(rcp85, rcp45)[1],  ks(rcp85, rcp60)[1], 0, ],
  ]
  )
p_result = pd.DataFrame(p_result, columns=['rcp26', 'rcp45', 'rcp60', 'rcp85'], index=['rcp26', 'rcp45', 'rcp60', 'rcp85'])

stop = ''