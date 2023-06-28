import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pandas as pd
import numpy as np

from SWCNT_Transition import *
from Spectral_Model import *
from globals_and_params import *
from OA_Fit import *
from data_retrieval import *

# New 1DF : 2.8, 20.0
# New 3DF : 1.9, 5.0

folder_in_cwd = "UV-Vis Nanotubes_04172017"#"OAdata3"
sample_name = "New 1DF.txt"#"(6,6) APTE DNA Paper.csv" #"metalrich.txt" #"(7,4) APTE DNA Paper.csv"

directory = os.path.dirname(os.path.realpath("__file__")) #os.path.dirname(os.path.join(os.cwd()))
mypath = os.path.join(directory, folder_in_cwd)
filename = os.path.join(mypath, sample_name)

data = retrieve_XY(filename)
data =  data[data[:,0]>360]
X = 1240./data[:,0]
Y = data[:,1]

a = Absorption_Spectra(E = X, absorbance = Y)
a.iter_nnls_fit()
# print(sorted(a.swcnt_soln_struct))
print("Total SWCNTs Concentration", a.swcnt_soln.sum(), "ug/ml")
print("Total Concentration", a.total_soln.sum(), "ug/ml")


fig = plt.figure()
plt.plot(X, Y, 'k')
plt.plot(X, a.total_model, 'r')
plt.plot(X, a.background_model, 'g')
plt.plot(X, a.swcnt_model, 'b')
# plt.plot(X, a.norm_abs(SWCNT(6,5)))

plt.show()
# for i in largest_nms:
#     idx = largest_nms[:,-2]
#     print swcnt_struct.T[i]
#     # print largest_nms.index
#     # print idx
    
#     plt.plot(X, swcnt_struct.ix[i])



# print(SWCNT(10,2).transitions[0].wp)
# print(SWCNT(8,1).transitions[1].wp)
# print({(n,m): 1240*i.Eii for n, m in nm_list() for i in SWCNT(n, m).transitions if 1030/1240.<i.Eii<1090/1240.})



#x, f, d, soln_struct, residual, combined_struct, full_model, swcnt_model = full_OA_fit(E=X, absorbance=Y)

#swcnts = soln_struct.groupby(nm_list())

# s = norm_SWCNT_struct()
# b = bkg_struct()
# all_absorption = pd.concat([bkg_struct(), norm_SWCNT_struct()], axis=1)
# sum_of_norm_semiconducting_swcnt = all_absorption.T.xs('S', level=1).sum()

#resid, combined_struct, swcnt_soln_struct, full_model, swcnt_model = NNLS_OA_fit(E=X, absorbance=Y)


# swcnt_struct = combined_struct.T.xs('SWCNT', level='Species Type').T #combined_struct.T.groupby('SWCNT', level='Species Type')#
# print swcnt_struct
# #swcnt = combined_struct[nm_list()]
# ##print soln_struct
# ##print swcnt
# ##print swcnt/np.sum(swcnt)



# swcnt_soln_struct.sort()#ascending=False)
# # print swcnt_soln_struct.T
# largest_nms = swcnt_soln_struct.T.index[:-6:-1]#ix('Species')[:-6:-1] # last 5 elements of reversed array
# percent_of_SWCNT = pd.Series(swcnt_soln_struct/swcnt_soln_struct.sum(), name='%')
# summary_all = pd.DataFrame([swcnt_soln_struct.T, percent_of_SWCNT.T], index=['Concentration (ug/ml)','%']).T
# # print summary_all
# # print largest_nms
# # print summary_all.ix[largest_nms]
