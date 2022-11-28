#!/usr/bin/env python3
import numpy as np
#import stats

import sklearn
from sklearn.neighbors import KernelDensity


import matplotlib.pyplot as plt
import sys
import subprocess
import scipy
from scipy import stats
from scipy.stats import norm
sys.path.append('../../../MESA/pyMESA/')
sys.path.append('../../../bulge_isochrones/')


# import isochrone_module as im 		# mine
import photometry_module as ph 		# mine

##########
bin_num = 'auto'
save_name = 'generative_KDE.pdf'
###########


data_file = 'stellar_ages.dat'
Joyce_ages, Bensby_ages= np.loadtxt(data_file, usecols=(0,1), unpack = True)


(Jmu, Jsigma) = norm.fit(Joyce_ages)
(Bmu, Bsigma) = norm.fit(Bensby_ages)

Jstats=r'$\mu=$' + "%.2f"%Jmu + ';'+r' $\sigma=$' + "%.2f"%Jsigma
Bstats=r'$\mu=$' + "%.2f"%Bmu + ';'+r' $\sigma=$' + "%.2f"%Bsigma


scale = len(Joyce_ages)


#np.histogram()
fig, ax = plt.subplots(figsize = (16,16))
ph.set_fig(ax)


n, bins, patches = plt.hist(Joyce_ages,  bins=bin_num, alpha = 1, color= 'navy',\
						    label = 'current work: MIST ages\nover basis of 0.5-20 Gyr')#+'\n'+Jstats)
#						    edgecolor = 'cornflowerblue', linewidth=5, linestyle = ':',\
y = scipy.stats.norm.pdf(bins, Jmu, Jsigma)*scale
plt.plot(bins, y, '--', color='cornflowerblue', linewidth=5, label='normal distribution: '+Jstats)


idx_array = np.linspace(min(Joyce_ages), max(Joyce_ages), 1000)
kde1 = stats.gaussian_kde(Joyce_ages)
y1 = kde1.pdf(idx_array)*scale


y1_resample  = kde1.resample(91)[0]
#print('y1_resample: ', y1_resample)
## the values are correct (5 - 17 Gyr) but the normalization is not
## --> in ax.hist, set density = False
x1_resample  = np.linspace(min(Joyce_ages), max(Joyce_ages), len(y1_resample)) ## make an array of the same length


kde2 = KernelDensity(bandwidth=0.1).fit(Joyce_ages.reshape(-1, 1))
y2 = kde2.sample(91)
#y2 = y2.ravel()
#y2 = y2*scale

kde3 = KernelDensity(bandwidth=0.01).fit(Joyce_ages.reshape(-1, 1))
y3 = kde3.sample(91)
#y3 = y3.ravel()
#y3 = y3*scale

# fig, ax = plt.subplots()
ax.plot(idx_array, y1, color='lightblue', label='KDE 1')
ax.hist(y1_resample, bins=8, density=False, color='lightblue', alpha=0.5, label=r'KDE 1 resampled; $h=auto$') ## set density = False
#ax.hist(y2, 		 bins=20, density=False, color='r', alpha=0.7, label=r'distribution drawn from $h = 0.1$')
ax.hist(y3, 		 bins=8, density=False, color='m', alpha=0.7, label=r'distribution drawn from $h = 0.01$')

plt.legend(loc=2, fontsize=14)

plt.xlabel('Ages (Gyr)', fontsize=30)
plt.ylabel('Count', fontsize=30)


#plt.show()


# plt.close()
# sys.exit()

# sample 44 new points from the data
#new_data = #kde.sample(91, random_state=0)
#new_data = pca.inverse_transform(new_data)

#print('new_data',new_data)

#ax.plot(idx_array, kde(idx_array), linewidth=5, linestyle='-', color='lightblue', label='KDE')


text_str = 'For ' + str(len(Joyce_ages) ) + ' targets of Bensby et al.\nAges from Teff, logg'

ax.annotate(text_str, xy=(1, 1), xytext=(-15, -15), fontsize=14,
xycoords='axes fraction', textcoords='offset points',
bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.65),\
horizontalalignment='right', verticalalignment='top') 


plt.savefig(save_name)

#plt.show()

plt.close()
subprocess.call('pdfcrop '+ save_name + ' ' + save_name, shell = True)

