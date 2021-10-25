import numpy as np
import math as m
from numba import jit
from multiprocessing import Pool
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

c = 2.
width, height = c*2.,c*4.
plt.rc('text', usetex=True)
fig = plt.figure(constrained_layout=True)
# fig = plt.figure(figsize = (width, height))
# constrained_layout=True
# plt.tight_layout()
gs = fig.add_gridspec(2,1)

f_ax1 = fig.add_subplot(gs[0, 0])
data_arrays= np.load('abp_Dr_data.npz')
Dr_data = []

for i in range(len(data_arrays.files)):
    label = 'arr_'+str(i)
    Dr_data.append(data_arrays[label])


Dr_data = np.asarray(Dr_data)

k_AB0, k_AB0_err,Drr_tab,t_obs,dt, k_AB, k_AB_err ,wbar_A, Abar_A, wbar_AB, Abar_AB,wbar_A_err, \
                Abar_A_err, wbar_AB_err, Abar_AB_err = Dr_data

vo = 1.4
Dtt = .5
Da = 1/((vo)**2./(6.*Drr_tab*Dtt))
Da = Drr_tab/Dtt


time = 2

plt.fill_between(Da, -0.5*Abar_AB[:,time], 2.1,facecolor="blue",color='blue', alpha=0.15)
plt.ylim(0.01,1.1)
plt.errorbar(Da, -0.5*Abar_AB[:,time],\
                yerr=0.5*wbar_AB_err[:,time],\
                fmt='o-', color = 'r',fillstyle='none', alpha = 1.,ms=8,label=r'  $\beta \langle Q \rangle_\lambda/2$')
plt.ylim(0.01,1.1)

plt.errorbar((Da), (np.log(k_AB)-np.log(0.00019587)),\
    yerr=((k_AB_err/k_AB)**2.)**0.5,fmt='d-', color = 'k',fillstyle='none',alpha = 1.,ms=8,\
    label=r'  $\ln k_\lambda/k_0$')
plt.ylim(0.01,1.1)

# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#            ncol=2, mode="expand", borderaxespad=0.,fontsize=20,frameon = False)
f_ax1.set_ylabel(r'  $\ln k_\lambda/k_0$', {'color': 'k', 'fontsize': 20})
f_ax1.tick_params(axis='y', color  = 'k', labelsize=16)

# f_ax1_twin = f_ax1.secondary_yaxis('right')
# secaxy.set_ylabel(r'$T\ [^oF]$')


f_ax1_twin = f_ax1.twinx()
f_ax1_twin.set_yticks([0.,.1,.2])
f_ax1_twin.set_ylabel(r'  $\beta \langle Q \rangle_\lambda/2$', {'color': 'r', 'fontsize': 20})
f_ax1_twin.yaxis.set_label_coords(1.035, 0.501)
# f_ax1_twin.yaxis.set_label_coords(1.3,0.501)
f_ax1_twin.tick_params(axis='y',color  = 'r', labelsize=16)
plt.setp(f_ax1_twin.get_yticklabels(), visible=False)



f_ax1.set_xlabel(r'  $D_\theta l^2_A/D_x$', {'color': 'k', 'fontsize': 20})
f_ax1.tick_params(axis='x',color  = 'k', labelsize=16)
# plt.xticks(size=16)
# yticks = f_ax1_twin.yaxis.get_major_ticks()
# yticks[-1].label1.set_visible(False)
# f_ax1_twin.tick_params(axis='y', labelcolor='r')


# f_ax1.plot(tt,x_eq ,'royalblue', alpha = 0.95,linewidth=1.)

# plt.xlabel(r'  $D_\theta l^2_A/D_x$', {'color': 'k', 'fontsize': 20})
plt.ylim(0.01,1.1)
plt.xlim(Da[0],Da[-1])
# plt.xlim(10**-1.,10**2.5)
plt.yscale('linear')
plt.xscale('log')
# plt.xticks(size=16)
# plt.yticks(size=16)
# plt.xlim([0,10000])
# plt.ylim([-1.45,1.45])
# plt.fill_between(tt, -1.35,-0.35,facecolor="grey", alpha=0.3)
a = 5.
plt.rc('text', usetex=True)
f_ax2 = fig.add_subplot(gs[1, 0])
data_arrays= np.load('abp_vo_data.npz')
vo_data = []

for i in range(len(data_arrays.files)):
    label = 'arr_'+str(i)
    vo_data.append(data_arrays[label])


k_AB0, k_AB0_err,vo_tab,t_obs,dt, k_AB, k_AB_err ,wbar_A, Abar_A, wbar_AB, Abar_AB,wbar_A_err, \
                Abar_A_err, wbar_AB_err, Abar_AB_err = np.asarray(vo_data)
# f_ax2.plot(range(10) ,'red', alpha = 0.9,linewidth=1.)

time = 3
xx = vo_tab/((16*a)/(3*(3)**0.5))
plt.errorbar(xx, (-0.5*Abar_AB[:,time]),\
yerr=Abar_AB_err[:,i],\
    fmt='o-', color = 'r',fillstyle='none', alpha = 0.95,ms=8)

plt.fill_between(xx, -0.5*Abar_AB[:,time], 10**1.5, facecolor="blue", color='blue',alpha=0.15)

plt.yscale('log')
plt.xscale('log')
plt.ylim(10**-1., 10**1.5)
plt.xlim(10**-1.7, 10**0)
plt.errorbar(vo_tab/((16*a)/(3*(3)**0.5)), (np.log(k_AB)-np.log(0.00019587)),\
     yerr=((k_AB_err/k_AB)**2.)**0.5,fmt='d-', color = 'k',fillstyle='none',alpha = 0.95,ms=8)

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
# plt.subplots_adjust(hspace=.2)

plt.show()
