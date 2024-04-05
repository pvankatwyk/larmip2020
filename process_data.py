import pandas as pd
import numpy as np

data = pd.read_csv(r'/oscar/scratch/pvankatw/datasets/sectors/AIS/with_chars/q0.5/train.csv')

larmip_models = [
  'VUB_AISMPALEO', 'NCAR_CISM', 'fETISh_16km', 'LSCE_GRISLI', 'IMAU_IMAUICE1', 'JPL1_ISSM', 
  'UCIJPL_ISSM', 'DOE_MALI', 'AWI_PISM1', 'PIK_PISM1', 'VUW_PISM', 'PIK_SICOPOLIS'
]

ismip_larmip_name_map = {
  'VUB_AISMPALEO': 'AISM_VUB',
  'NCAR_CISM': 'CISM_NCA',
  'fETISh_16km': 'FETI_VUB',
  'LSCE_GRISLI': 'GRIS_LSC',
  'IMAU_IMAUICE1': 'IMAU_VUB',
  'JPL1_ISSM': 'ISSM_JPL',
  'UCIJPL_ISSM': 'ISSM_UCI',
  'DOE_MALI': 'MALI_LAN',
  'AWI_PISM1': 'PISM_AWI',
  'PIK_PISM1': 'PISM_PIK',
  'VUW_PISM': 'PISM_VUW',
  'PIK_SICOPOLIS': 'SICO_UHO'
  
}
data = data[data.model.isin(larmip_models)]
data['larmip_model'] = data.model.map(ismip_larmip_name_map)


def ReadOceanScaling(fname):
  # from https://github.com/radical-collaboration/facts/blob/main/modules/larmip/AIS/larmip_icesheet_fit.py
	# Initialize return variables
	nodelay = []
	widelay = []
	delay = []

	# Read in the ocean scaling from the file
	with open(fname) as f:
		for line in f:
			lp = line.split("\t")
			nodelay.append(float(lp[0]))
			widelay.append(float(lp[3]))
			delay.append(float(lp[2]))

	# Convert to numpy arrays
	nodelay = np.array(nodelay)
	widelay = np.array(widelay)
	delay = np.array(delay)

	return(nodelay, widelay, delay)

def get_scaling_models(scaling_coefficients_dir="./larmip/ScalingCoefficients/"):
  
    NumOmodel = 19
    OS_NoDelay_R1, OS_WiDelay_R1, OS_Delay_R1 = ReadOceanScaling(f"{scaling_coefficients_dir}/OceanScaling/OS_R1.dat")
    OS_NoDelay_R2, OS_WiDelay_R2, OS_Delay_R2 = ReadOceanScaling(f"{scaling_coefficients_dir}/OceanScaling/OS_R2.dat")
    OS_NoDelay_R3, OS_WiDelay_R3, OS_Delay_R3 = ReadOceanScaling(f"{scaling_coefficients_dir}/OceanScaling/OS_R3.dat")
    OS_NoDelay_R4, OS_WiDelay_R4, OS_Delay_R4 = ReadOceanScaling(f"{scaling_coefficients_dir}/OceanScaling/OS_R4.dat")

    # Read melting sensitivity
    fname = f"{scaling_coefficients_dir}/MeltSensitivity/MeltSensitivity.dat" # File to read
    with open(fname) as f:
      MeltSensitivity = np.array([float(row) for row in f])

    # Save the calibration data to a pickle
    output = {"OS_NoDelay_R1": OS_NoDelay_R1, "OS_WiDelay_R1": OS_WiDelay_R1, "OS_Delay_R1": OS_Delay_R1, \
        "OS_NoDelay_R2": OS_NoDelay_R2, "OS_WiDelay_R2": OS_WiDelay_R2, "OS_Delay_R2": OS_Delay_R2, \
        "OS_NoDelay_R3": OS_NoDelay_R3, "OS_WiDelay_R3": OS_WiDelay_R3, "OS_Delay_R3": OS_Delay_R3, \
        "OS_NoDelay_R4": OS_NoDelay_R4, "OS_WiDelay_R4": OS_WiDelay_R4, "OS_Delay_R4": OS_Delay_R4, \
        "MeltSensitivity": MeltSensitivity, "NumOmodel": NumOmodel}
    
    return output

def read_response_functions(model_name, response_function_dir="./larmip/RFunctions/", tlen=None):
    
	# Read in the RF from the files
	fname = f"{response_function_dir}/RF_{model_name}_BM08_R1.dat"
	with open(fname) as f:
		r1 = np.array([float(row) for row in f])

	fname = f"{response_function_dir}/RF_{model_name}_BM08_R2.dat"
	with open(fname) as f:
		r2 = np.array([float(row) for row in f])

	fname = f"{response_function_dir}/RF_{model_name}_BM08_R3.dat"
	with open(fname) as f:
		r3 = np.array([float(row) for row in f])

	fname = f"{response_function_dir}/RF_{model_name}_BM08_R4.dat"
	with open(fname) as f:
		r4 = np.array([float(row) for row in f])

	fname = f"{response_function_dir}/RF_{model_name}_BM08_R5.dat"
	with open(fname) as f:
		r5 = np.array([float(row) for row in f])
  
  	# Pad with zeros if necessary
	if tlen is not None:

		zerofill = np.zeros([tlen-200])
		r1 = np.concatenate((r1,zerofill))
		r2 = np.concatenate((r2,zerofill))
		r3 = np.concatenate((r3,zerofill))
		r4 = np.concatenate((r4,zerofill))
		r5 = np.concatenate((r5,zerofill))

	# Done
	return r1, r2, r3, r4, r5
    
stop = ''