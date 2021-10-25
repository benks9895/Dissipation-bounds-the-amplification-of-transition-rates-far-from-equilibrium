import numpy as np
import math as m
from numba import jit
from multiprocessing import Pool
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
plt.rc('text', usetex=True)

a=7.
f = 1.4
tot_steps = 1*10**7
dt = 0.005
t_relax = 1/(8.*a)**0.5
w_index1,w_index2,w_index3 = np.array([15,37,55])
t_obs1,t_obs2,t_obs3 = np.array([500,800,400])


def get_inst_data(f,w,t_obs):
    # data_arrays= np.load('duff_inst_f_'+str(f)+'_w_'+str(w)+'_t_obs_'+str(t_obs)+'.npz', allow_pickle = True)
    data_arrays = np.load('duff_inst_f_'+str(f)+'_w_'+str(w)+'_t_obs_'+str(t_obs)+'.npz', allow_pickle = True)
    inst_data = data_arrays['arr_0']
    print('duff_inst_f_'+str(f)+'_w_'+str(w)+'_t_obs_'+str(t_obs)+'.npz')
    # inst_data = []
    #
    # for i in range(len(data_arrays.files)):
    #     label = 'arr_'+str(i)
    #     inst_data.append(data_arrays[label])

    # print(np.shape(inst_data))
    # np.savez_compressed('duff_inst_f_'+str(f)+'_w_'+str(w)+'_t_obs_'+str(t_obs), inst_data)
    inst_data = np.asarray(inst_data)
    t_obs, dt,x_inst, w_inst, A_inst,phi_inst,lo, wAB, AAB,phiAB,wA,AA,phiA = inst_data
    t_tab = np.linspace(-t_obs*dt,t_obs*dt,2*t_obs )

    inst_data_small = [t_tab, w_inst[:200], A_inst[:200],phi_inst[:200], wAB, AAB,phiAB,wA,AA,phiA]
    fname = 'duff_inst_small_f_'+str(f)+'_w_'+str(w)+'_t_obs_'+str(t_obs)
    np.save(fname,inst_data_small)

    return t_tab,x_inst, w_inst, A_inst,phi_inst,lo, wAB, AAB,phiAB,wA,AA,phiA

data_arrays= np.load('duff_omega_data.npz')
# w_tab,t_obs,dt, k_AB, k_AB_err ,wbar_A, Abar_A, wbar_AB, Abar_AB,wbar_A_err, \
#                 Abar_A_err, wbar_AB_err, Abar_AB_err = omega_data
omega_data = []
for i in range(len(data_arrays.files)):
    label = 'arr_'+str(i)
    omega_data.append(data_arrays[label])
omega_data = np.asarray(omega_data)
w_tab = omega_data[0]
k_AB_tab = omega_data[3]
k_AB0 = k_AB_tab[0]
wAB_tab = omega_data[7]


fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2,3)

w_rel  = np.array([w_tab[w_index1],w_tab[w_index2],w_tab[w_index3]])/(8*(7-1.4)-.24)**0.5
# print('w_rel = ',w_rel)
# subplot 1 up
f_ax1_up = fig.add_subplot(gs[0, 0])
w,t_obs = w_tab[w_index1], t_obs1
# t_tab,x_inst, w_inst, A_inst,phi_inst,lo, wAB, AAB,phiAB,wA,AA,phiA = get_inst_data(f,w,t_obs)
fname = 'duff_inst_small_f_'+str(f)+'_w_'+str(w)+'_t_obs_'+str(t_obs)+'.npy'
inst_data_small = np.load(fname, allow_pickle = True)
t_tab, w_inst, A_inst,phi_inst, wAB, AAB,phiAB,wA,AA,phiA = inst_data_small
t_tab = t_tab/t_relax
kamp = np.log(k_AB_tab[w_index1]/k_AB_tab[0])
# print('kamp = ',kamp)
ttm = (t_tab-t_obs)*dt
# Q(t)
print(len(w_inst))
for i in range(200):
    plt.plot(t_tab,w_inst[i],'b',alpha = 0.025,ms=8)
plt.plot(t_tab,wAB,'r-',fillstyle='none',alpha = 0.95,ms=8)

# plt.plot(t_tab,wA,'r--',fillstyle='none',alpha = 0.95)
# plt.plot(t_tab,(wAB-wA),'k--',alpha = 0.95)
plt.axhline(y=kamp,color ='k')
plt.xticks(size=16)
plt.yticks(size=16)
# plt.plot(t_tab,wAB, 'k--')
plt.xlim( t_tab[0], t_tab[-1])
# plt.ylim(-kamp, 2*kamp)
plt.ylabel(r'  $\beta \langle \delta(x(0))Q \rangle_\lambda/2$', {'color': 'k', 'fontsize': 20})
plt.xlabel(r'  $\tau/\tau_A$', {'color': 'k', 'fontsize': 20})
plt.xlim(t_tab[0], t_tab[-1])
plt.ylim(-2, 5)


# subplot 1 low
f_ax1_low = fig.add_subplot(gs[1, 0])

# A(t)
for i in range(200):
    plt.plot(t_tab,A_inst[i]-phi_inst[i],'b',alpha = 0.025,ms=8)

# plt.plot(t_tab,AAB,'r-',fillstyle='none',alpha = 0.95,ms=8)
# plt.plot(t_tab,AA,'r--',alpha = 0.95)
# plt.plot(t_tab,(AAB-AA),'k--',alpha = 0.95)
# plt.plot(t_tab,phiAB,'r--',alpha = 0.95)
plt.plot(t_tab,(AAB-1.*phiAB),'r-',alpha = 0.95)
plt.axhline(y=kamp,color ='k')
plt.xticks(size=16)
plt.yticks(size=16)
plt.ylabel(r'   $\beta \langle \delta(x(0)) \Gamma \rangle_\lambda/2$', {'color': 'k', 'fontsize': 20})
plt.xlabel(r'  $\tau/\tau_A$', {'color': 'k', 'fontsize': 20})
plt.xlim(t_tab[0], t_tab[-1])
plt.ylim(-2, 5)




# subplot 2 up
f_ax2_up = fig.add_subplot(gs[0, 1])
w,t_obs = w_tab[w_index2], t_obs2
# t_tab,x_inst, w_inst, A_inst,phi_inst,lo, wAB, AAB,phiAB,wA,AA,phiA = get_inst_data(f,w,t_obs)
fname = 'duff_inst_small_f_'+str(f)+'_w_'+str(w)+'_t_obs_'+str(t_obs)+'.npy'
inst_data_small = np.load(fname, allow_pickle = True)
t_tab, w_inst, A_inst,phi_inst, wAB, AAB,phiAB,wA,AA,phiA = inst_data_small
t_tab = t_tab/t_relax
kamp = np.log(k_AB_tab[w_index2]/k_AB_tab[0])
# print('kamp = ',kamp)
ttm = (t_tab-t_obs)*dt
# Q(t)
for i in range(200):
    plt.plot(t_tab,w_inst[i],'b',alpha = 0.025,ms=8)
plt.plot(t_tab,wAB,'r-',fillstyle='none',alpha = 0.95,ms=8)
# plt.plot(t_tab,wA,'r--',fillstyle='none',alpha = 0.95)
# plt.plot(t_tab,(wAB-wA),'k--',alpha = 0.95)
plt.axhline(y=kamp,color ='k')
plt.xticks(size=16)
plt.yticks(size=16)
# plt.plot(t_tab,wAB, 'k--')
plt.xlim( t_tab[0], t_tab[-1])
# plt.ylim(-kamp, 2*kamp)
plt.ylabel(r'  $\beta \langle \delta(x(0)) Q \rangle_\lambda/2$', {'color': 'k', 'fontsize': 20})
plt.xlabel(r'  $\tau/\tau_A$', {'color': 'k', 'fontsize': 20})
plt.xlim(t_tab[0], t_tab[-1])
plt.ylim(-2, 5)


# subplot 2 low
f_ax2_low = fig.add_subplot(gs[1, 1])

# A(t)
for i in range(200):
    plt.plot(t_tab,A_inst[i]-phi_inst[i],'b',alpha = 0.025,ms=8)

# plt.plot(t_tab,AAB,'r-',fillstyle='none',alpha = 0.95,ms=8)
# plt.plot(t_tab,AA,'r--',alpha = 0.95)
# plt.plot(t_tab,(AAB-AA),'k--',alpha = 0.95)
# plt.plot(t_tab,phiAB,'r--',alpha = 0.95)
plt.plot(t_tab,(AAB-1.*phiAB),'r-',alpha = 0.95)
plt.axhline(y=kamp,color ='k')
plt.xticks(size=16)
plt.yticks(size=16)
plt.ylabel(r'   $\beta \langle \delta(x(0)) \Gamma \rangle_\lambda/2$', {'color': 'k', 'fontsize': 20})
plt.xlabel(r'  $\tau/\tau_A$', {'color': 'k', 'fontsize': 20})
plt.xlim(t_tab[0], t_tab[-1])
plt.ylim(-2, 5)




# subplot 3
f_ax3_up = fig.add_subplot(gs[0, 2])
w,t_obs = w_tab[w_index3], t_obs3
# t_tab,x_inst, w_inst, A_inst,phi_inst,lo, wAB, AAB,phiAB,wA,AA,phiA = get_inst_data(f,w,t_obs)
fname = 'duff_inst_small_f_'+str(f)+'_w_'+str(w)+'_t_obs_'+str(t_obs)+'.npy'
inst_data_small = np.load(fname, allow_pickle = True)
t_tab, w_inst, A_inst,phi_inst, wAB, AAB,phiAB,wA,AA,phiA = inst_data_small
t_tab = t_tab/t_relax
kamp = np.log(k_AB_tab[w_index3]/k_AB_tab[0])
# print('kamp = ',kamp)
ttm = (t_tab-t_obs)*dt
# Q(t)
for i in range(200):
    plt.plot(t_tab,w_inst[i],'b',alpha = 0.025,ms=8)
plt.plot(t_tab,wAB,'r-',fillstyle='none',alpha = 0.95,ms=8)
# plt.plot(t_tab,wA,'r--',fillstyle='none',alpha = 0.95)
# plt.plot(t_tab,(wAB-wA),'k--',alpha = 0.95)
plt.axhline(y=kamp,color ='k')
plt.xticks(size=16)
plt.yticks(size=16)
# plt.plot(t_tab,wAB, 'k--')
plt.xlim( t_tab[0], t_tab[-1])
plt.ylim(-2, 5)
# plt.ylim(-kamp, 2*kamp)
plt.ylabel(r'  $\beta \langle \delta(x(0)) Q \rangle_\lambda/2$', {'color': 'k', 'fontsize': 20})
plt.xlabel(r'  $\tau/\tau_A$', {'color': 'k', 'fontsize': 20})


# subplot 3 low
f_ax3_low = fig.add_subplot(gs[1, 2])

# A(t)
for i in range(200):
    plt.plot(t_tab,A_inst[i]-phi_inst[i],'b',alpha = 0.025,ms=8)

# plt.plot(t_tab,AAB,'r-',fillstyle='none',alpha = 0.95,ms=8)
# plt.plot(t_tab,AA,'r--',alpha = 0.95)
# plt.plot(t_tab,(AAB-AA),'k--',alpha = 0.95)
# plt.plot(t_tab,phiAB,'r--',alpha = 0.95)
plt.plot(t_tab,(AAB-1.*phiAB),'r-',alpha = 0.95)
plt.axhline(y=kamp,color ='k')
plt.xticks(size=16)
plt.yticks(size=16)
plt.ylabel(r'   $\beta \langle \delta(x(0)) \Gamma \rangle_\lambda/2$', {'color': 'k', 'fontsize': 20})
plt.xlabel(r'  $\tau/\tau_A$', {'color': 'k', 'fontsize': 20})
plt.xlim(t_tab[0], t_tab[-1])

plt.ylim(-2, 5)



plt.show()
