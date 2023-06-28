import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#def gaussian(x, center, width):
#    return np.exp(-(x - center)**2 / (2 * width**2))

#x = np.linspace(1,10,1000)
#mesh = np.meshgrid(x,x)
#f = gaussian(x, 0, 3)
#g = gaussian(x, 0, 5)
#Gauss2d = np.outer(f,g)
#plt.imshow(Gauss2d, interpolation='nearest', cmap=cm.gist_rainbow)
#plt.show()


import pandas as pd
import numpy as np
from SWCNT_Transition import *
from Spectral_Model import *
from globals_and_params import *


Emm_65 = norm_PL(SWCNT(6,5))
print(Emm_65)
plt.imshow(Emm_65, extent=[Emm_65.index[0], Emm_65.index[-1], Emm_65.index[0], Emm_65.index[-1]], interpolation='nearest', cmap=cm.gist_rainbow)
plt.show()

#from mpl_toolkits.mplot3d import Axes3D
##from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pylab import *

X_Abs = norm_SWCNT_struct()
#X_Abs = transition_struct(SWCNT(6,5))
Abs_65 = X_Abs['(6,5)']#.loc[1:2.001]['(6,5)']
QY_65 = X_Abs['(6,5)']#.loc[2:3]['(6,5)']
#Abs_65 = X_Abs['22(0)']#X_Abs.loc[1:2.001]['22(0)']
#QY_65 = X_Abs['22(0)']#X_Abs.loc[2:3]['22(0)']
print((Abs_65.values.shape))
print((QY_65.values.shape))
Emm_65 = np.outer(Abs_65.values, QY_65.values)
print(Emm_65)


fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(Abs_65.index, AQY_65.index, Emm_65, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#hist2d(Abs_65.index, QY_65.index, bins=40, norm=LogNorm())
plt.imshow(Emm_65, extent=[Abs_65.index[0], Abs_65.index[-1], QY_65.index[0], QY_65.index[-1]], interpolation='nearest', cmap=cm.gist_rainbow)
plt.show()
##default_a = 1
##default_a_min = 0
##default_a_max = 2
##default_b = 2
##default_b_min = 1
##default_b_max = 3
##
##params = pd.DataFrame.from_csv('Default_params.csv')
###print(params.loc['Value'])
###print(np.array(params.loc['Value']['alpha']))
###print(SWCNT(6,5).dt)
##print(Transition(6,5, 3).Eii)
###print(pd.DataFrame.from_csv('Default_params.csv'))
##def default_param_struct():
##    a = np.array([default_a, default_a_min, default_a_max])
##    b = np.array([default_b, default_b_min, default_b_max])
##    d = {'a': a, 'b': b}
##    return pd.DataFrame(d, index=['val', 'min','max'])
##
##def f(a=default_a, b=default_b, *args, **kwargs):
##    return kwargs
##
##default_param_df = default_param_struct()
###print default_param_df
##def_param_dict = default_param_df.loc['val'].to_dict()
###print def_param_dict


# This should print {'a': 3, 'b': 2}  (i.e. a is passed and default_b is given automatically)
#print f({'a':3})
# This should print {'a': 1, 'b': 2}  (as this is the default parameter structure calculated in the function)
#print f(def_param_dict)
# This should print {'a': 1, 'b': 2}  (default parameter structure is assumed)
#print f()
# This should print {'a': 4, 'b': 2}  (i.e. a is passed and default_b is given automatically)
#print f(a=4)
# This should print {'a': 3, 'b': 5}  as they are both passed
#print f(a=3, b=5)


