import scipy.io
import numpy as np

# ECG
data_ecg = scipy.io.loadmat("ECG.mat")

for i in data_ecg:
	if '__' not in i and 'readme' not in i:
		np.savetxt(("ECG/"+i+".csv"),data_ecg[i],delimiter=' ',fmt='%s')

# JAP
data_JAP = scipy.io.loadmat("JAP.mat")

for i in data_ecg:
	if '__' not in i and 'readme' not in i:
		np.savetxt(("JAP/"+i+".csv"),data_JAP[i],delimiter=' ',fmt='%s')

# PHAL
data_PHAL = scipy.io.loadmat("PHAL.mat")

for i in data_PHAL:
	if '__' not in i and 'readme' not in i:
		np.savetxt(("PHAL/"+i+".csv"),data_PHAL[i],delimiter=' ',fmt='%s')

# WAF
data_WAF = scipy.io.loadmat("WAF.mat")

for i in data_WAF:
	if '__' not in i and 'readme' not in i:
		np.savetxt(("WAF/"+i+".csv"),data_WAF[i],delimiter=' ',fmt='%s')