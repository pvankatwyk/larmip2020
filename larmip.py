import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import random as rnd
import json

# # Load in Atmospheric Forcing (SAT) for the years 1900-2100


# Read atmospheric forcing

# NumTensemble = 600
# Tlen = 651

# #fname = "../MAGICC/RCP2500/Temp_HistConstraint.dat" # File to read
# #df = pd.read_csv(fname,sep='\s+',index_col=0,header=None)
# #df.columns.name = "ensemble member"
# #df.index.name = "Time"
# #T = np.array(df.values)

# fname = "../MAGICC/RCP2500/Temp_HistConstraint.dat" # File to read
# df = pd.read_csv(fname,sep='\s+',index_col=0,header=None)
# df.columns.name = "ensemble member"
# df.index.name = "Time"
# SAT = np.array(df.values)

# print(len(SAT[:,1]))
# # SAT[time,ensemblemember]


# # Normalize and crop temperature series
# Temp = []
# Tavebeg = 0
# Taveend = 80
# tbeg = 51 #1901
# tend = 251 #2101
# for i in range(len(SAT[1,:])):
#     SATave = np.mean(SAT[Tavebeg:Taveend,i])
#     SAT[:,i] = SAT[:,i]-SATave
# SAT = SAT[tbeg:tend,:]


# model = 'AISM_VUB'
# data = pd.read_csv(r'/oscar/scratch/pvankatw/datasets/sectors/AIS/with_chars/q0.5/train.csv')


# data = data[data.model.isin(larmip_models)]
# data['larmip_model'] = data.model.map(ismip_larmip_name_map)

# model_runs = data.loc[data.larmip_model == model].reset_index(drop=True)
# test_case = model_runs.loc[model_runs.id == np.random.choice(model_runs.id.unique())]
# GSAT = test_case.ts_anomaly
# sle = test_case.sle / 1000

# Load the scaler
# scaler_path = "/users/pvankatw/research/larmip2020/scaler_y.pkl"
# with open(scaler_path, 'rb') as f:
#     scaler = pickle.load(f)

# # Unscaled the variable sle
# sle = scaler.inverse_transform(sle.values.reshape(-1, 1)).flatten()

def run_all_larmip(magicc_dir="./larmip/MAGICC/RCP2500/", save=True, num_iters=100):
    scaling_coefficients = get_scaling_models()
    results = {}
    for scenario in ['RCP26', 'RCP45', 'RCP60', 'RCP85']:
        print(f'\n--- SCENARIO {scenario} ---')
        results[scenario] = {}
        SAT = load_magicc_data(magicc_dir, scenario)
        for model in larmip_models:
            print('Model:', model)
            model_result = project(SAT, model, scaling_coefficients, num_iters=num_iters)
            results[scenario][model] = model_result
            
    if save:
        with open('results.json', 'w') as f:
            json.dump(results, f)
    return results

def load_magicc_data(magicc_dir, scenario):
    fname = f"{magicc_dir}/Temp_{scenario}_HistConstraint.dat" # File to read
    df = pd.read_csv(fname,sep='\s+',index_col=0,header=None)
    df.columns.name = "ensemble member"
    df.index.name = "Time"
    SAT = np.array(df.values)
    
    Temp = []
    Tavebeg = 0
    Taveend = 80
    tbeg = 51 #1901
    tend = 251 #2101
    for i in range(len(SAT[1,:])):
        SATave = np.mean(SAT[Tavebeg:Taveend,i])
        SAT[:,i] = SAT[:,i]-SATave
    SAT = SAT[tbeg:tend,:]
    
    return SAT

def project(GSAT, model_name, scaling_coefficients=None,  num_iters=100):
    
    if scaling_coefficients is None:
        scaling_coefficients = get_scaling_models()

    MeltSensitivity = scaling_coefficients['MeltSensitivity']
    
    NumTensemble = 600
    tbeg = 51 #1901
    tend = 251 #2101
    NumOmodel = 19
    
    RF_BM08_R1, RF_BM08_R2, RF_BM08_R3, RF_BM08_R4, RF_BM08_R5 = read_response_functions(model_name=model_name)


    EnsembleSize = num_iters
    countR1, countR2, countR3, countR4 = 0, 0, 0, 0

    # preallocate memory
    projection_length = tend - tbeg
    SL_wTd_nos_base_R1 = [0] * (projection_length)
    SL_wTd_nos_base_R2 = [0] * (projection_length)
    SL_wTd_nos_base_R3 = [0] * (projection_length)
    SL_wTd_nos_base_R4 = [0] * (projection_length)
    SL_wTd_nos_base = [0] * (projection_length)

    for _ in range(EnsembleSize):

        # Select forcing randomly

        # SELECT FORCINGS FROM RANDOM ENSEMBLE MEMBERS
        # select global warming path
        iTens = rnd.randint(0,NumTensemble-1)
        temperature_array = np.array(GSAT[:,iTens])

        # RANDOMLY SELECT AN OCEAN MODEL
        iOmod = rnd.randint(0,NumOmodel-1)
        OS_R1 = scaling_coefficients['OS_WiDelay_R1'][iOmod]
        OS_R2 = scaling_coefficients['OS_WiDelay_R2'][iOmod]
        OS_R3 = scaling_coefficients['OS_WiDelay_R3'][iOmod]
        OS_R4 = scaling_coefficients['OS_WiDelay_R4'][iOmod]
        OS_R5 = scaling_coefficients['OS_WiDelay_R4'][iOmod]

        tau_R1 = int(scaling_coefficients['OS_Delay_R1'][iOmod])
        tau_R2 = int(scaling_coefficients['OS_Delay_R2'][iOmod])
        tau_R3 = int(scaling_coefficients['OS_Delay_R3'][iOmod])
        tau_R4 = int(scaling_coefficients['OS_Delay_R4'][iOmod])
        tau_R5 = int(scaling_coefficients['OS_Delay_R4'][iOmod])

        if tau_R1>0:
            countR1 = countR1+1
        if tau_R2>0:
            countR2 = countR2+1
        if tau_R3>0:
            countR3 = countR3+1
        if tau_R4>0:
            countR4 = countR4+1
        
        
        # STOPPED HERE -- NEED TO KEEP GOING THROUGH AND MAKING CHANGES
        Temp_R1 = np.append(np.zeros(tau_R1),temperature_array[:projection_length-tau_R1])
        Temp_R2 = np.append(np.zeros(tau_R2),temperature_array[:projection_length-tau_R2])
        Temp_R3 = np.append(np.zeros(tau_R3),temperature_array[:projection_length-tau_R3])
        Temp_R4 = np.append(np.zeros(tau_R4),temperature_array[:projection_length-tau_R4])
        Temp_R5 = np.append(np.zeros(tau_R5),temperature_array[:projection_length-tau_R5])
        
        # select melting sensitivity
        MS_R1 = rnd.uniform(MeltSensitivity[0],MeltSensitivity[1])
        MS_R2 = rnd.uniform(MeltSensitivity[0],MeltSensitivity[1])
        MS_R3 = rnd.uniform(MeltSensitivity[0],MeltSensitivity[1])
        MS_R4 = rnd.uniform(MeltSensitivity[0],MeltSensitivity[1])
        MS_R5 = rnd.uniform(MeltSensitivity[0],MeltSensitivity[1])

        # COMPOSE TIME SERIES AS A FUNCTION OF MELT SENSITIVITY, OCEAN MODEL, AND SAT
        # Compose forcing time series
        M_R1 = MS_R1*OS_R1*Temp_R1
        M_R2 = MS_R2*OS_R2*Temp_R2
        M_R3 = MS_R3*OS_R3*Temp_R3
        M_R4 = MS_R4*OS_R4*Temp_R4
        M_R5 = MS_R5*OS_R5*Temp_R5

        M_R1[M_R1 < 0.0] = 0.0
        M_R2[M_R2 < 0.0] = 0.0
        M_R3[M_R3 < 0.0] = 0.0
        M_R4[M_R4 < 0.0] = 0.0
        M_R5[M_R5 < 0.0] = 0.0

        # Linear response
        SL = []
        SL.append(0)
        for t in range(1,len(temperature_array)):
            #print(t)
            dSL = 0
            for tp in range(0,t):
                #print(t,tp)
                dSL = dSL + M_R1[tp]*RF_BM08_R1[t-tp]
            SL.append(dSL)
        SL_wTd_nos_base_R1=np.vstack([SL_wTd_nos_base_R1, SL])

        SL = []
        SL.append(0)
        for t in range(1,len(temperature_array)):
            #print(t)
            dSL = 0
            for tp in range(0,t):
                #print(t,tp)
                dSL = dSL + M_R2[tp]*RF_BM08_R2[t-tp]
            SL.append(dSL)
        SL_wTd_nos_base_R2=np.vstack([SL_wTd_nos_base_R2, SL])

        SL = []
        SL.append(0)
        for t in range(1,len(temperature_array)):
            #print(t)
            dSL = 0
            for tp in range(0,t):
                #print(t,tp)
                dSL = dSL + M_R3[tp]*RF_BM08_R3[t-tp]
            SL.append(dSL)
        SL_wTd_nos_base_R3=np.vstack([SL_wTd_nos_base_R3, SL])

        SL = []
        SL.append(0)
        for t in range(1,len(temperature_array)):
            #print(t)
            dSL = 0
            for tp in range(0,t):
                #print(t,tp)
                dSL = dSL + M_R4[tp]*RF_BM08_R4[t-tp]
            SL.append(dSL)
        SL_wTd_nos_base_R4=np.vstack([SL_wTd_nos_base_R4, SL])

        SL = []
        SL.append(0)
        for t in range(1,len(temperature_array)):
            #print(t)
            dSL = 0
            for tp in range(0,t):
                #print(t,tp)
                dSL = dSL + M_R5[tp]*RF_BM08_R5[t-tp]
            SL.append(dSL)
        SL_wTd_nos_base=np.vstack([SL_wTd_nos_base, SL])

    SL_wTd_nos_base_SU = SL_wTd_nos_base_R1+SL_wTd_nos_base_R2+SL_wTd_nos_base_R3+SL_wTd_nos_base_R4+SL_wTd_nos_base


    return SL_wTd_nos_base_SU

def plot_larmip_results(SL_wTd_nos_base_SU, model_name, scenario, true=None):



        Time = np.arange(2015,2101)


        SL_wTd_nos_base_SU_50pc = np.percentile(SL_wTd_nos_base_SU, 50, axis=0, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
        SL_wTd_nos_base_SU_83pc = np.percentile(SL_wTd_nos_base_SU, 83.33, axis=0, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
        SL_wTd_nos_base_SU_17pc = np.percentile(SL_wTd_nos_base_SU, 16.66, axis=0, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
        SL_wTd_nos_base_SU_95pc = np.percentile(SL_wTd_nos_base_SU, 5, axis=0, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
        SL_wTd_nos_base_SU_05pc = np.percentile(SL_wTd_nos_base_SU, 95, axis=0, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
        SL_wTd_nos_base_SU_99pc = np.percentile(SL_wTd_nos_base_SU, 99, axis=0, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
        SL_wTd_nos_base_SU_01pc = np.percentile(SL_wTd_nos_base_SU, 1, axis=0, out=None, overwrite_input=False, interpolation='linear', keepdims=False)



        # fp1 = plt.figure()
        # plt.plot(Time,SL_wTd_nos_base_SU_50pc,'k')
        # plt.plot(Time,SL_wTd_nos_base_SU_83pc,'m')
        # plt.plot(Time,SL_wTd_nos_base_SU_17pc,'--m')
        # plt.plot(Time,SL_wTd_nos_base_SU_95pc,'r')
        # plt.plot(Time,SL_wTd_nos_base_SU_05pc,'--r')
        # plt.plot(Time,SL_wTd_nos_base_SU_99pc,'y')
        # plt.plot(Time,SL_wTd_nos_base_SU_01pc,'--y')


        # fp1.savefig("SL_wTd_nos_base_AISM_VUB_SU_percentiles_lines.png", bbox_inches='tight')



        fp2 = plt.figure()
        plt.fill_between(Time, SL_wTd_nos_base_SU_01pc, SL_wTd_nos_base_SU_99pc,facecolor='#ffff00')
        plt.fill_between(Time, SL_wTd_nos_base_SU_05pc, SL_wTd_nos_base_SU_95pc,facecolor='#ff8800')
        plt.fill_between(Time, SL_wTd_nos_base_SU_17pc, SL_wTd_nos_base_SU_83pc,facecolor='#ff0000')
        plt.plot(Time, SL_wTd_nos_base_SU_50pc, 'k-')
        if true is not None:
            plt.plot(Time, true, 'g-')
        plt.show()
        fp2.savefig(f"./plots/SL_wTd_nos_base_{model_name}_SU_{scenario}_percentiles_shades.png", bbox_inches='tight')

        stop = ''


def project_ismip(GSAT, model_name, scaling_coefficients=None, num_iters=100, region=None):
    
    if scaling_coefficients is None:
        scaling_coefficients = get_scaling_models()

    MeltSensitivity = scaling_coefficients['MeltSensitivity']
    
    NumTensemble = 1
    tbeg = 0 #2015
    tend = 86 #2101
    NumOmodel = 19
    GSAT = np.array(GSAT)
    GSATave = np.mean(GSAT)
    GSAT = GSAT-GSATave
    
    
    RF_BM08_R1, RF_BM08_R2, RF_BM08_R3, RF_BM08_R4, RF_BM08_R5 = read_response_functions(model_name=model_name)


    EnsembleSize = num_iters
    countR1, countR2, countR3, countR4 = 0, 0, 0, 0

    # preallocate memory
    projection_length = tend - tbeg
    SL_wTd_nos_base_R1 = [0] * (projection_length)
    SL_wTd_nos_base_R2 = [0] * (projection_length)
    SL_wTd_nos_base_R3 = [0] * (projection_length)
    SL_wTd_nos_base_R4 = [0] * (projection_length)
    SL_wTd_nos_base_R5 = [0] * (projection_length)

    for _ in range(EnsembleSize):

        # Select forcing randomly

        # SELECT FORCINGS FROM RANDOM ENSEMBLE MEMBERS
        # select global warming path
        temperature_array = np.array(GSAT)

        # RANDOMLY SELECT AN OCEAN MODEL
        iOmod = rnd.randint(0,NumOmodel-1)
        OS_R1 = scaling_coefficients['OS_WiDelay_R1'][iOmod]
        OS_R2 = scaling_coefficients['OS_WiDelay_R2'][iOmod]
        OS_R3 = scaling_coefficients['OS_WiDelay_R3'][iOmod]
        OS_R4 = scaling_coefficients['OS_WiDelay_R4'][iOmod]
        OS_R5 = scaling_coefficients['OS_WiDelay_R4'][iOmod]

        tau_R1 = int(scaling_coefficients['OS_Delay_R1'][iOmod])
        tau_R2 = int(scaling_coefficients['OS_Delay_R2'][iOmod])
        tau_R3 = int(scaling_coefficients['OS_Delay_R3'][iOmod])
        tau_R4 = int(scaling_coefficients['OS_Delay_R4'][iOmod])
        tau_R5 = int(scaling_coefficients['OS_Delay_R4'][iOmod])

        if tau_R1>0:
            countR1 = countR1+1
        if tau_R2>0:
            countR2 = countR2+1
        if tau_R3>0:
            countR3 = countR3+1
        if tau_R4>0:
            countR4 = countR4+1
        
        
        # STOPPED HERE -- NEED TO KEEP GOING THROUGH AND MAKING CHANGES
        Temp_R1 = np.append(np.zeros(tau_R1),temperature_array[:projection_length-tau_R1])
        Temp_R2 = np.append(np.zeros(tau_R2),temperature_array[:projection_length-tau_R2])
        Temp_R3 = np.append(np.zeros(tau_R3),temperature_array[:projection_length-tau_R3])
        Temp_R4 = np.append(np.zeros(tau_R4),temperature_array[:projection_length-tau_R4])
        Temp_R5 = np.append(np.zeros(tau_R5),temperature_array[:projection_length-tau_R5])
        
        # select melting sensitivity
        MS_R1 = rnd.uniform(MeltSensitivity[0],MeltSensitivity[1])
        MS_R2 = rnd.uniform(MeltSensitivity[0],MeltSensitivity[1])
        MS_R3 = rnd.uniform(MeltSensitivity[0],MeltSensitivity[1])
        MS_R4 = rnd.uniform(MeltSensitivity[0],MeltSensitivity[1])
        MS_R5 = rnd.uniform(MeltSensitivity[0],MeltSensitivity[1])

        # COMPOSE TIME SERIES AS A FUNCTION OF MELT SENSITIVITY, OCEAN MODEL, AND SAT
        # Compose forcing time series
        M_R1 = MS_R1*OS_R1*Temp_R1
        M_R2 = MS_R2*OS_R2*Temp_R2
        M_R3 = MS_R3*OS_R3*Temp_R3
        M_R4 = MS_R4*OS_R4*Temp_R4
        M_R5 = MS_R5*OS_R5*Temp_R5

        M_R1[M_R1 < 0.0] = 0.0
        M_R2[M_R2 < 0.0] = 0.0
        M_R3[M_R3 < 0.0] = 0.0
        M_R4[M_R4 < 0.0] = 0.0
        M_R5[M_R5 < 0.0] = 0.0

        # Linear response
        SL = []
        SL.append(0)
        for t in range(1,len(temperature_array)):
            #print(t)
            dSL = 0
            for tp in range(0,t):
                #print(t,tp)
                dSL = dSL + M_R1[tp]*RF_BM08_R1[t-tp]
            SL.append(dSL)
        SL_wTd_nos_base_R1=np.vstack([SL_wTd_nos_base_R1, SL])

        SL = []
        SL.append(0)
        for t in range(1,len(temperature_array)):
            #print(t)
            dSL = 0
            for tp in range(0,t):
                #print(t,tp)
                dSL = dSL + M_R2[tp]*RF_BM08_R2[t-tp]
            SL.append(dSL)
        SL_wTd_nos_base_R2=np.vstack([SL_wTd_nos_base_R2, SL])

        SL = []
        SL.append(0)
        for t in range(1,len(temperature_array)):
            #print(t)
            dSL = 0
            for tp in range(0,t):
                #print(t,tp)
                dSL = dSL + M_R3[tp]*RF_BM08_R3[t-tp]
            SL.append(dSL)
        SL_wTd_nos_base_R3=np.vstack([SL_wTd_nos_base_R3, SL])

        SL = []
        SL.append(0)
        for t in range(1,len(temperature_array)):
            #print(t)
            dSL = 0
            for tp in range(0,t):
                #print(t,tp)
                dSL = dSL + M_R4[tp]*RF_BM08_R4[t-tp]
            SL.append(dSL)
        SL_wTd_nos_base_R4=np.vstack([SL_wTd_nos_base_R4, SL])

        SL = []
        SL.append(0)
        for t in range(1,len(temperature_array)):
            #print(t)
            dSL = 0
            for tp in range(0,t):
                #print(t,tp)
                dSL = dSL + M_R5[tp]*RF_BM08_R5[t-tp]
            SL.append(dSL)
        SL_wTd_nos_base_R5=np.vstack([SL_wTd_nos_base_R5, SL])

    SL_wTd_nos_base_SU = SL_wTd_nos_base_R1+SL_wTd_nos_base_R2+SL_wTd_nos_base_R3+SL_wTd_nos_base_R4+SL_wTd_nos_base_R5

    if region is None or region.lower() in ('sum', 'all'):
        return SL_wTd_nos_base_SU
    elif region == 1 or region == 'R1':
        return SL_wTd_nos_base_R1
    elif region == 2 or region == 'R2':
        return SL_wTd_nos_base_R2
    elif region == 3 or region == 'R3':
        return SL_wTd_nos_base_R3
    elif region == 4 or region == 'R4':
        return SL_wTd_nos_base_R4
    elif region == 5 or region == 'R5':
        return SL_wTd_nos_base_R5
    else:
        raise ValueError("Region must be one of 'sum', 'all', or an integer between 1 and 5")
    
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

def read_response_functions(model_name, response_function_dir="./larmip/RFunctions/", ):
    
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

	# for 2015 - 2100
	# r1 = r1[-86:]
	# r2 = r2[-86:]
	# r3 = r3[-86:]
	# r4 = r4[-86:]
	# r5 = r5[-86:]

	return r1, r2, r3, r4, r5


# data = pd.read_csv(r'/oscar/scratch/pvankatw/datasets/sectors/AIS/with_chars/q0.5/train.csv')

ismip_models = [
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

larmip_models = [ismip_larmip_name_map[model] for model in ismip_models]

ismip_sectors_to_larmip_regions = {
	1: 'R3',
	2: 'R2',
	3: 'R3',
	4: 'R3',
	5: 'R4',
	6: 'R2',
 	7: 'R1',
	8: 'R1',
	9: 'R1',
 	10: 'R1',
	11: 'R4',
	12: 'R1',
 	13: 'R1',
	14: 'R1',
	15: 'R4',
 	16: 'R5',
	17: 'R5',
	18: 'R5',
}



    