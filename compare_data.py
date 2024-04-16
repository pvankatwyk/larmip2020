import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ismip = pd.read_csv('withclim_False_train_with_gsat.csv')
ismip = ismip[ismip.Scenario == 'rcp8.5']
ismip = ismip['gsat']

magicc = pd.read_csv(r'/users/pvankatw/research/larmip2020/larmip/MAGICC/RCP2500/Temp_RCP85_HistConstraint.dat',sep='\s+',index_col=0,header=None)
magicc = pd.DataFrame(magicc.iloc[51:251].values.flatten(), columns=['gsat'])
magicc.index.name = "Time"


# Plot
plt.figure(figsize=(10, 6))
plt.hist(ismip, bins=30, alpha=0.5, label='ISMIP', density=True, color='blue')
plt.hist(magicc['gsat'], bins=50, alpha=0.5, label='MAGICC', density=True, color='red')

# Adding labels and title
plt.xlabel('Global Surface Air Temperature (°C)')
plt.ylabel('Probability Density')
plt.title('Normalized Histogram of Global Surface Air Temperatures')
plt.legend()
plt.tight_layout()

plt.savefig('gsat_histogram.png')
# Show the plot
plt.show()
stop = ''