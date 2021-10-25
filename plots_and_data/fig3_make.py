import numpy as np
import math as m
from numba import jit
from multiprocessing import Pool
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

a=7.
plt.rc('text', usetex=True)
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2,1)

print('test')
f_ax1 = fig.add_subplot(gs[0, 0])




data_arrays= np.load('duff_omega_data.npz')
omega_data = []

for i in range(len(data_arrays.files)):
    label = 'arr_'+str(i)
    omega_data.append(data_arrays[label])


omega_data = np.asarray(omega_data)

w_tab,t_obs,dt, k_AB, k_AB_err ,wbar_A, Abar_A, wbar_AB, Abar_AB,wbar_A_err, \
                Abar_A_err, wbar_AB_err, Abar_AB_err = omega_data

wAB, wAB_err = np.zeros(len(w_tab)),np.zeros(len(w_tab))
time = np.array([-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -15, -15, -15, -15, -15,
       -15, -15, -25, -25, -25, -25, -25, -25, -30, -35, -41, -45,-49,-51, -53, -54, -55, 25, 25, 25, 25, 25,
       25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
       25, 25, 25, 25, 25, 25])
for i in range(len(w_tab)):
    wAB[i],wAB_err[i] = wbar_AB[i,time[i]],wbar_AB_err[i,time[i]]

wres = (8.*(a-1.4)-0.25)**.5
wrel = w_tab/wres
plt.errorbar(wrel, 0.5*wAB ,\
        yerr=wAB_err,\
        fmt='o-', color = 'r',fillstyle='none', alpha = 0.95,ms=8,label=r'  $\beta \langle Q \rangle_\lambda/2$')



plt.fill_between(wrel, 0.5*wAB, 3.5,facecolor="blue",color='blue',alpha=0.15)


plt.errorbar(wrel, np.log(k_AB)-np.log(k_AB[0]),\
        yerr=((k_AB_err/k_AB)**2.+(k_AB_err[0]/k_AB[0])**2.)**0.5
                 \
        ,fmt='d-', color = 'k',fillstyle='none',\
        alpha = 0.95,ms=8,label=r'  $\ln k_\lambda/k_0$')
plt.ylim(0,3.)
# plt.legend(loc='top',fontsize=20)
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#            ncol=2, mode="expand", borderaxespad=0.,fontsize=20,frameon = False)

f_ax1.set_ylabel(r'  $\ln k_\lambda/k_0$', {'color': 'k', 'fontsize': 20})
f_ax1.tick_params(axis='y', color  = 'k', labelsize=16)

# f_ax1_twin = f_ax1.secondary_yaxis('right')
# secaxy.set_ylabel(r'$T\ [^oF]$')


f_ax1_twin = f_ax1.twinx()
# f_ax1_twin.set_yticks([0.,.1,.2])
f_ax1_twin.set_ylabel(r'  $\beta \langle Q \rangle_\lambda/2$', {'color': 'r', 'fontsize': 20})
f_ax1_twin.yaxis.set_label_coords(1.035, 0.501)
# f_ax1_twin.yaxis.set_label_coords(1.3,0.501)
f_ax1_twin.tick_params(axis='y',color  = 'r', labelsize=16)
plt.setp(f_ax1_twin.get_yticklabels(), visible=False)



f_ax1.set_xlabel(r'  $$\omega/\omega^*$$', {'color': 'k', 'fontsize': 20})
f_ax1.tick_params(axis='x',color  = 'k', labelsize=16)

plt.xlim(wrel[0],wrel[-1])
plt.xlim(.1,1.9)
plt.xticks(color='k', size=16,alpha =1. )
plt.yticks(size=16)


# plt.xlabel(r' $\mathrm{driving~frequency}~\omega/\omega^*$', {'color': 'k', 'fontsize': 20})

f_ax2 = fig.add_subplot(gs[1, 0])
# plt.rc('text', usetex=True)

data_arrays= np.load('duff_f_data_2.npz')
f_data = []

for i in range(len(data_arrays.files)):
    label = 'arr_'+str(i)
    f_data.append(data_arrays[label])



f_data = np.asarray(f_data)

w_tab,f_tab,t_obs,dt, k_AB, k_AB_err ,wbar_A, Abar_A, wbar_AB, Abar_AB,wbar_A_err, \
                Abar_A_err, wbar_AB_err, Abar_AB_err = f_data

# f_ax2.plot(range(10) ,'red', alpha = 0.9,linewidth=1.)
plt.rc('text', usetex=True)
time = 24
wA, wAB, AA, AAB = wbar_A[:,time],wbar_AB[:,time],Abar_A[:,time],Abar_AB[:,time]
xx = f_tab/((8*a)/(3*(3)**0.5))
plt.errorbar(xx, np.abs(.5*wAB),\
                         yerr=wbar_AB_err[:,time],\
                         \
                         fmt='o-', color = 'r',fillstyle='none', alpha = 1.,ms=8)

plt.fill_between(f_tab/((8*a)/(3*(3)**0.5)),np.abs(.5*wAB), 10**(1.5), facecolor="blue", color='blue',alpha=0.15)



plt.errorbar(f_tab/((8*a)/(3*(3)**0.5)), np.log(k_AB)-np.log(k_AB[0]),\
                 yerr=((k_AB_err/k_AB)**2.+(k_AB_err[0]/k_AB[0])**2.)**0.5\
                 \
                 ,fmt='kd-', fillstyle='none',alpha = .95,ms=8)

plt.ylim(10**(-2.),10**(1.5))
plt.xlim(10**(-2.),1.25)
plt.yscale('log')
plt.xscale('log')
plt.xticks(size=16)
plt.yticks(size=16)

plt.xlabel(r'  $f/F_\mathrm{m}$', {'color': 'k', 'fontsize': 20, 'fontweight':'bold'})
f_ax2.set_ylabel(r'  $\ln k_\lambda/k_0$', {'color': 'k', 'fontsize': 20})
f_ax2.tick_params(axis='y', color  = 'k', labelsize=16)
f_ax2_twin = f_ax2.twinx()
f_ax2_twin.set_ylabel(r'  $\beta \langle Q \rangle_\lambda/2$', {'color': 'r', 'fontsize': 20})
f_ax2_twin.yaxis.set_label_coords(1.035, 0.501)
f_ax2_twin.tick_params(axis='y',color  = 'r', labelsize=16)
plt.setp(f_ax2_twin.get_yticklabels(), visible=False)
f_ax2.tick_params(axis='x',color  = 'k', labelsize=16)
# plt.xlabel(r'  $f/|F^\mathrm{eq}_\mathrm{max}|$', {'color': 'k', 'fontsize': 20, 'fontweight':'bold'})
plt.show()
