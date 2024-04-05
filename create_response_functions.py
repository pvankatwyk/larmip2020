
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import imp
import csv
import pandas as pd
from io import StringIO

# AISM_VUB = AISM_VUB

larmip_data_folder = "./larmip/CreateRFunctions/larmip_data/"
# Experiment BM02

fname="larmip/CreateRFunctions/larmip_data/AISM_VUB/2m.nc"
ncf = nc.Dataset(fname, "r")
ncf.variables.keys()

AISM_VUB_BM02_R0 = ncf.variables["All regions forced together"][:]
AISM_VUB_BM02_R1 = ncf.variables["EAIS"][:]
AISM_VUB_BM02_R2 = ncf.variables["Ross Sea"][:]
AISM_VUB_BM02_R3 = ncf.variables["Amundsen Sea"][:]
AISM_VUB_BM02_R4 = ncf.variables["Weddell Sea"][:]
AISM_VUB_BM02_R5 = ncf.variables["Peninsula"][:]

AISM_VUB_BM02_SU = AISM_VUB_BM02_R1+AISM_VUB_BM02_R2+AISM_VUB_BM02_R3+AISM_VUB_BM02_R4+AISM_VUB_BM02_R5

# create and write response function
RF_AISM_VUB_BM02_R0=np.diff(AISM_VUB_BM02_R0)/2
RF_AISM_VUB_BM02_R1=np.diff(AISM_VUB_BM02_R1)/2
RF_AISM_VUB_BM02_R2=np.diff(AISM_VUB_BM02_R2)/2
RF_AISM_VUB_BM02_R3=np.diff(AISM_VUB_BM02_R3)/2
RF_AISM_VUB_BM02_R4=np.diff(AISM_VUB_BM02_R4)/2
RF_AISM_VUB_BM02_R5=np.diff(AISM_VUB_BM02_R5)/2
RF_AISM_VUB_BM02_SU=np.diff(AISM_VUB_BM02_SU)/2

RF_AISM_VUB_BM02_R0=np.append(RF_AISM_VUB_BM02_R0[0],RF_AISM_VUB_BM02_R0)
RF_AISM_VUB_BM02_R1=np.append(RF_AISM_VUB_BM02_R1[0],RF_AISM_VUB_BM02_R1)
RF_AISM_VUB_BM02_R2=np.append(RF_AISM_VUB_BM02_R2[0],RF_AISM_VUB_BM02_R2)
RF_AISM_VUB_BM02_R3=np.append(RF_AISM_VUB_BM02_R3[0],RF_AISM_VUB_BM02_R3)
RF_AISM_VUB_BM02_R4=np.append(RF_AISM_VUB_BM02_R4[0],RF_AISM_VUB_BM02_R4)
RF_AISM_VUB_BM02_R5=np.append(RF_AISM_VUB_BM02_R5[0],RF_AISM_VUB_BM02_R5)
RF_AISM_VUB_BM02_SU=np.append(RF_AISM_VUB_BM02_SU[0],RF_AISM_VUB_BM02_SU)


# plot
plt.plot(AISM_VUB_BM02_R0,label="AISM_VUB_BM02_R0")
plt.plot(AISM_VUB_BM02_R1,label="AISM_VUB_BM02_R1")
plt.plot(AISM_VUB_BM02_R2,label="AISM_VUB_BM02_R2")
plt.plot(AISM_VUB_BM02_R3,label="AISM_VUB_BM02_R3")
plt.plot(AISM_VUB_BM02_R4,label="AISM_VUB_BM02_R4")
plt.plot(AISM_VUB_BM02_R5,label="AISM_VUB_BM02_R5")
plt.plot(AISM_VUB_BM02_SU,label="AISM_VUB_BM02_SU")
plt.legend()

print(len(RF_AISM_VUB_BM02_R5))

# AISM_VUB = AISM_VUB
# Experiment BM04

fname="larmip_data/AISM_VUB/4m.nc"
ncf = nc.Dataset(fname, "r")
ncf.variables.keys()

AISM_VUB_BM04_R0 = ncf.variables["All regions forced together"][:]
AISM_VUB_BM04_R1 = ncf.variables["EAIS"][:]
AISM_VUB_BM04_R2 = ncf.variables["Ross Sea"][:]
AISM_VUB_BM04_R3 = ncf.variables["Amundsen Sea"][:]
AISM_VUB_BM04_R4 = ncf.variables["Weddell Sea"][:]
AISM_VUB_BM04_R5 = ncf.variables["Peninsula"][:]

AISM_VUB_BM04_SU = AISM_VUB_BM04_R1+AISM_VUB_BM04_R2+AISM_VUB_BM04_R3+AISM_VUB_BM04_R4+AISM_VUB_BM04_R5

# create and write response function
RF_AISM_VUB_BM04_R0=np.diff(AISM_VUB_BM04_R0)/4
RF_AISM_VUB_BM04_R1=np.diff(AISM_VUB_BM04_R1)/4
RF_AISM_VUB_BM04_R2=np.diff(AISM_VUB_BM04_R2)/4
RF_AISM_VUB_BM04_R3=np.diff(AISM_VUB_BM04_R3)/4
RF_AISM_VUB_BM04_R4=np.diff(AISM_VUB_BM04_R4)/4
RF_AISM_VUB_BM04_R5=np.diff(AISM_VUB_BM04_R5)/4
RF_AISM_VUB_BM04_SU=np.diff(AISM_VUB_BM04_SU)/4

RF_AISM_VUB_BM04_R0=np.append(RF_AISM_VUB_BM04_R0[0],RF_AISM_VUB_BM04_R0)
RF_AISM_VUB_BM04_R1=np.append(RF_AISM_VUB_BM04_R1[0],RF_AISM_VUB_BM04_R1)
RF_AISM_VUB_BM04_R2=np.append(RF_AISM_VUB_BM04_R2[0],RF_AISM_VUB_BM04_R2)
RF_AISM_VUB_BM04_R3=np.append(RF_AISM_VUB_BM04_R3[0],RF_AISM_VUB_BM04_R3)
RF_AISM_VUB_BM04_R4=np.append(RF_AISM_VUB_BM04_R4[0],RF_AISM_VUB_BM04_R4)
RF_AISM_VUB_BM04_R5=np.append(RF_AISM_VUB_BM04_R5[0],RF_AISM_VUB_BM04_R5)
RF_AISM_VUB_BM04_SU=np.append(RF_AISM_VUB_BM04_SU[0],RF_AISM_VUB_BM04_SU)


# plot
plt.plot(AISM_VUB_BM04_R0,label="All")
plt.plot(AISM_VUB_BM04_R1,label="East Antarctica")
plt.plot(AISM_VUB_BM04_R2,label="Ross Sea")
plt.plot(AISM_VUB_BM04_R3,label="Amundsen Sea")
plt.plot(AISM_VUB_BM04_R4,label="Weddell Sea")
plt.plot(AISM_VUB_BM04_R5,label="Larsen")
plt.plot(AISM_VUB_BM04_SU,label="AISM_VUB_BM04_SU")
plt.legend()

#plt.figure()
#plt.plot(AISM_VUB_BM04_R0,label="All")
#plt.plot(AISM_VUB_BM04_R2,label="Ross Sea")
#plt.plot(AISM_VUB_BM04_R4,label="Weddell Sea")
#plt.plot(AISM_VUB_BM04_SU,label="AISM_VUB_BM04_SU")
#plt.legend()

#plt.figure()
#plt.plot(AISM_VUB_BM04_R1,label="East Antarctica")
#plt.plot(AISM_VUB_BM04_R3,label="Amundsen Sea")
#plt.plot(AISM_VUB_BM04_R5,label="Larsen")
#plt.legend()

print(len(RF_AISM_VUB_BM04_R5))


# AISM_VUB = AISM_VUB
# Experiment BM08

fname="larmip_data/AISM_VUB/8m.nc"
ncf = nc.Dataset(fname, "r")
ncf.variables.keys()

AISM_VUB_BM08_R0 = ncf.variables["All regions forced together"][:]
AISM_VUB_BM08_R1 = ncf.variables["EAIS"][:]
AISM_VUB_BM08_R2 = ncf.variables["Ross Sea"][:]
AISM_VUB_BM08_R3 = ncf.variables["Amundsen Sea"][:]
AISM_VUB_BM08_R4 = ncf.variables["Weddell Sea"][:]
AISM_VUB_BM08_R5 = ncf.variables["Peninsula"][:]

AISM_VUB_BM08_SU = AISM_VUB_BM08_R1+AISM_VUB_BM08_R2+AISM_VUB_BM08_R3+AISM_VUB_BM08_R4+AISM_VUB_BM08_R5

# create and write response function
RF_AISM_VUB_BM08_R0=np.diff(AISM_VUB_BM08_R0)/8
RF_AISM_VUB_BM08_R1=np.diff(AISM_VUB_BM08_R1)/8
RF_AISM_VUB_BM08_R2=np.diff(AISM_VUB_BM08_R2)/8
RF_AISM_VUB_BM08_R3=np.diff(AISM_VUB_BM08_R3)/8
RF_AISM_VUB_BM08_R4=np.diff(AISM_VUB_BM08_R4)/8
RF_AISM_VUB_BM08_R5=np.diff(AISM_VUB_BM08_R5)/8
RF_AISM_VUB_BM08_SU=np.diff(AISM_VUB_BM08_SU)/8

RF_AISM_VUB_BM08_R0=np.append(RF_AISM_VUB_BM08_R0[0],RF_AISM_VUB_BM08_R0)
RF_AISM_VUB_BM08_R1=np.append(RF_AISM_VUB_BM08_R1[0],RF_AISM_VUB_BM08_R1)
RF_AISM_VUB_BM08_R2=np.append(RF_AISM_VUB_BM08_R2[0],RF_AISM_VUB_BM08_R2)
RF_AISM_VUB_BM08_R3=np.append(RF_AISM_VUB_BM08_R3[0],RF_AISM_VUB_BM08_R3)
RF_AISM_VUB_BM08_R4=np.append(RF_AISM_VUB_BM08_R4[0],RF_AISM_VUB_BM08_R4)
RF_AISM_VUB_BM08_R5=np.append(RF_AISM_VUB_BM08_R5[0],RF_AISM_VUB_BM08_R5)
RF_AISM_VUB_BM08_SU=np.append(RF_AISM_VUB_BM08_SU[0],RF_AISM_VUB_BM08_SU)


# plot
plt.plot(AISM_VUB_BM08_R0,label="All")
plt.plot(AISM_VUB_BM08_R1,label="East Antarctica")
plt.plot(AISM_VUB_BM08_R2,label="Ross Sea")
plt.plot(AISM_VUB_BM08_R3,label="Amundsen Sea")
plt.plot(AISM_VUB_BM08_R4,label="Weddell Sea")
plt.plot(AISM_VUB_BM08_R5,label="Larsen")
#plt.plot(AISM_VUB_BM08_SU,label="AISM_VUB_BM08_SU")
plt.legend()

print(len(RF_AISM_VUB_BM08_R5))


# Response functions
np.savetxt("RF_AISM_VUB_BM02_R0.dat", RF_AISM_VUB_BM02_R0, delimiter=",")
np.savetxt("RF_AISM_VUB_BM02_R1.dat", RF_AISM_VUB_BM02_R1, delimiter=",")
np.savetxt("RF_AISM_VUB_BM02_R2.dat", RF_AISM_VUB_BM02_R2, delimiter=",")
np.savetxt("RF_AISM_VUB_BM02_R3.dat", RF_AISM_VUB_BM02_R3, delimiter=",")
np.savetxt("RF_AISM_VUB_BM02_R4.dat", RF_AISM_VUB_BM02_R4, delimiter=",")
np.savetxt("RF_AISM_VUB_BM02_R5.dat", RF_AISM_VUB_BM02_R5, delimiter=",")
np.savetxt("RF_AISM_VUB_BM02_SU.dat", RF_AISM_VUB_BM02_SU, delimiter=",")


np.savetxt("RF_AISM_VUB_BM04_R0.dat", RF_AISM_VUB_BM04_R0, delimiter=",")
np.savetxt("RF_AISM_VUB_BM04_R1.dat", RF_AISM_VUB_BM04_R1, delimiter=",")
np.savetxt("RF_AISM_VUB_BM04_R2.dat", RF_AISM_VUB_BM04_R2, delimiter=",")
np.savetxt("RF_AISM_VUB_BM04_R3.dat", RF_AISM_VUB_BM04_R3, delimiter=",")
np.savetxt("RF_AISM_VUB_BM04_R4.dat", RF_AISM_VUB_BM04_R4, delimiter=",")
np.savetxt("RF_AISM_VUB_BM04_R5.dat", RF_AISM_VUB_BM04_R5, delimiter=",")
np.savetxt("RF_AISM_VUB_BM04_SU.dat", RF_AISM_VUB_BM04_SU, delimiter=",")


np.savetxt("RF_AISM_VUB_BM08_R0.dat", RF_AISM_VUB_BM08_R0, delimiter=",")
np.savetxt("RF_AISM_VUB_BM08_R1.dat", RF_AISM_VUB_BM08_R1, delimiter=",")
np.savetxt("RF_AISM_VUB_BM08_R2.dat", RF_AISM_VUB_BM08_R2, delimiter=",")
np.savetxt("RF_AISM_VUB_BM08_R3.dat", RF_AISM_VUB_BM08_R3, delimiter=",")
np.savetxt("RF_AISM_VUB_BM08_R4.dat", RF_AISM_VUB_BM08_R4, delimiter=",")
np.savetxt("RF_AISM_VUB_BM08_R5.dat", RF_AISM_VUB_BM08_R5, delimiter=",")
np.savetxt("RF_AISM_VUB_BM08_SU.dat", RF_AISM_VUB_BM08_SU, delimiter=",")


# Experiments
np.savetxt("AISM_VUB_BM02_R0.dat", AISM_VUB_BM02_R0, delimiter=",")
np.savetxt("AISM_VUB_BM02_R1.dat", AISM_VUB_BM02_R1, delimiter=",")
np.savetxt("AISM_VUB_BM02_R2.dat", AISM_VUB_BM02_R2, delimiter=",")
np.savetxt("AISM_VUB_BM02_R3.dat", AISM_VUB_BM02_R3, delimiter=",")
np.savetxt("AISM_VUB_BM02_R4.dat", AISM_VUB_BM02_R4, delimiter=",")
np.savetxt("AISM_VUB_BM02_R5.dat", AISM_VUB_BM02_R5, delimiter=",")
np.savetxt("AISM_VUB_BM02_SU.dat", AISM_VUB_BM02_SU, delimiter=",")


np.savetxt("AISM_VUB_BM04_R0.dat", AISM_VUB_BM04_R0, delimiter=",")
np.savetxt("AISM_VUB_BM04_R1.dat", AISM_VUB_BM04_R1, delimiter=",")
np.savetxt("AISM_VUB_BM04_R2.dat", AISM_VUB_BM04_R2, delimiter=",")
np.savetxt("AISM_VUB_BM04_R3.dat", AISM_VUB_BM04_R3, delimiter=",")
np.savetxt("AISM_VUB_BM04_R4.dat", AISM_VUB_BM04_R4, delimiter=",")
np.savetxt("AISM_VUB_BM04_R5.dat", AISM_VUB_BM04_R5, delimiter=",")
np.savetxt("AISM_VUB_BM04_SU.dat", AISM_VUB_BM04_SU, delimiter=",")


np.savetxt("AISM_VUB_BM08_R0.dat", AISM_VUB_BM08_R0, delimiter=",")
np.savetxt("AISM_VUB_BM08_R1.dat", AISM_VUB_BM08_R1, delimiter=",")
np.savetxt("AISM_VUB_BM08_R2.dat", AISM_VUB_BM08_R2, delimiter=",")
np.savetxt("AISM_VUB_BM08_R3.dat", AISM_VUB_BM08_R3, delimiter=",")
np.savetxt("AISM_VUB_BM08_R4.dat", AISM_VUB_BM08_R4, delimiter=",")
np.savetxt("AISM_VUB_BM08_R5.dat", AISM_VUB_BM08_R5, delimiter=",")
np.savetxt("AISM_VUB_BM08_SU.dat", AISM_VUB_BM08_SU, delimiter=",")





