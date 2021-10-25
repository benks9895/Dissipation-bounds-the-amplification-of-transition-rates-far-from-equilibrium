import numpy as np
import math as m
from numba import jit
from multiprocessing import Pool
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
plt.rc('text', usetex=True)

maxtraj = 200
a=5.
vo = 1.
tot_steps = 1*10**7
dt = 0.005
t_relax = 1/(8.*a)**0.5
Drr_index1,Drr_index2,Drr_index3 = 7,55,70
t_obs1,t_obs2,t_obs3 = 250,150,250


def get_inst_data(vo,Drr,t_obs, dt):
    # print('loading data!!!')
    print('abp_inst_vo_'+str(vo)+'_Drr_'+str(Drr)+'_t_obs_'+str(t_obs)+'_dt_'+str(dt)+'.npz')
    data_arrays= np.load('abp_inst_vo_'+str(vo)+'_Drr_'+str(Drr)+'_t_obs_'+str(t_obs)+'_dt_'+str(dt)+'.npz')
    inst_data = []

    for i in range(len(data_arrays.files)):
        label = 'arr_'+str(i)
        inst_data.append(data_arrays[label])


    print(np.shape(inst_data))
    # inst_data = np.asarray(inst_data)
    # for i in range(len(inst_data)):
    #     print(inst_data[i])
    t_obs, dt,x_inst, w_inst, A_inst,phi_inst,lo, wAB, AAB,phiAB,wA,AA,phiA = inst_data

    t_tab = np.linspace(-t_obs*dt,t_obs*dt,2*t_obs )
    inst_data_small = [t_tab, w_inst[:200], A_inst[:200],phi_inst[:200], wAB, AAB,phiAB,wA,AA,phiA]
    fname = 'abp_inst_small_vo_'+str(vo)+'_Drr_'+str(Drr)+'_t_obs_'+str(t_obs)+'_dt_'+str(dt)
    np.save(fname,inst_data_small)
    return t_tab, x_inst, w_inst, A_inst,phi_inst,lo, wAB, AAB,phiAB,wA,AA,phiA

data_arrays= np.load('abp_Dr_data.npz')
print('abp_Dr_data.npz')
Dr_data = []
for i in range(len(data_arrays.files)):
    label = 'arr_'+str(i)
    Dr_data.append(data_arrays[label])
Dr_data = np.asarray(Dr_data)
# k_AB0, k_AB0_err,Drr_tab,t_obs,dt, k_AB, k_AB_err ,wbar_A, Abar_A, wbar_AB, Abar_AB,wbar_A_err, \
#                 Abar_A_err, wbar_AB_err, Abar_AB_err = Dr_data
Drr_tab = Dr_data[2]
k_AB0 = 0.00019587
k_AB_tab = Dr_data[5]
k_AB0 = 0.00019587
k_AB_err_tab = Dr_data[6]
AAB_tab = Dr_data[10]

# print('Dr/Dr12 = ',  np.array([Drr_tab[Drr_index1],Drr_tab[Drr_index2],Drr_tab[Drr_index3]])/Drr_tab[Drr_index2])

# print(np.log(k_AB_tab/k_AB0))


fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2,3)


# subplot 1 up
f_ax1_up = fig.add_subplot(gs[1, 0])
Drr,t_obs = Drr_tab[Drr_index1], t_obs1
print(Drr)
# t_tab, x_inst, w_inst, A_inst,phi_inst,lo, wAB, AAB,phiAB,wA,AA,phiA = get_inst_data(vo,Drr,t_obs, dt)
fname = 'abp_inst_small_vo_'+str(vo)+'_Drr_'+str(Drr)+'_t_obs_'+str(t_obs)+'_dt_'+str(dt)+'.npy'
inst_data_small = np.load(fname, allow_pickle = True)
t_tab, w_inst, A_inst,phi_inst, wAB, AAB,phiAB,wA,AA,phiA = inst_data_small
t_tab = t_tab/t_relax

print(np.log(k_AB_tab[Drr_index1]/k_AB0))
kamp = np.log(k_AB_tab/k_AB0)
# print('kamp = ',kamp)
ttm = (t_tab-t_obs)*dt
# A(t)
for i in range(maxtraj):
    plt.plot(t_tab,w_inst[i],'b',alpha = 0.025,ms=8)

# plt.plot(t_tab,wAB,'r',alpha = 0.95)
# plt.plot(t_tab,2.*phiAB,'r--',alpha = 0.95)
plt.plot(t_tab,(wAB-2.*phiAB),'r',alpha = 0.95,ms=8)
plt.axhline(y=kamp[Drr_index1],color ='k',alpha = 0.95,ms=8)
plt.ylabel(r'  $\beta \langle \delta(x(0)) \Gamma \rangle_\lambda/2$', {'color': 'k', 'fontsize': 20})
plt.xlabel(r'  $\tau/\tau_A$', {'color': 'k', 'fontsize': 20})
plt.xlim(t_tab[0], t_tab[-1])
plt.ylim(-3,3)
plt.xticks(size=16)
plt.yticks(size=16)
# subplot 1 low
f_ax1_low = fig.add_subplot(gs[0, 0])

# Q(t)
for i in range(maxtraj):
    plt.plot(t_tab,A_inst[i],'b',alpha = 0.025,ms=8)

plt.plot(t_tab,AAB,'r',alpha = 0.95,ms=8)
# plt.plot(t_tab,phiAB,'r--',alpha = 0.95)
# plt.plot(t_tab,(AAB-phiAB),'k--',alpha = 0.95)
plt.axhline(y=kamp[Drr_index1],color ='k',alpha = 0.95,ms=8)
plt.ylabel(r'  $\beta \langle \delta(x(0)) Q \rangle_\lambda/2$', {'color': 'k', 'fontsize': 20})
plt.xlabel(r'  $\tau/\tau_A$', {'color': 'k', 'fontsize': 20})
plt.xticks(size=16)
plt.yticks(size=16)




# subplot 2 up
f_ax2_up = fig.add_subplot(gs[1, 1])
Drr,t_obs = Drr_tab[Drr_index2], t_obs2
print(Drr)
# t_tab, x_inst, w_inst, A_inst,phi_inst,lo, wAB, AAB,phiAB,wA,AA,phiA = get_inst_data(vo,Drr,t_obs, dt)
fname = 'abp_inst_small_vo_'+str(vo)+'_Drr_'+str(Drr)+'_t_obs_'+str(t_obs)+'_dt_'+str(dt)+'.npy'
inst_data_small = np.load(fname, allow_pickle = True)
t_tab, w_inst, A_inst,phi_inst, wAB, AAB,phiAB,wA,AA,phiA = inst_data_small
t_tab = t_tab/t_relax
# print('kamp = ',kamp)
ttm = (t_tab-t_obs)*dt
# A(t)
for i in range(maxtraj):
    plt.plot(t_tab,w_inst[i],'b',alpha = 0.025,ms=8)

# plt.plot(t_tab,wAB,'r',alpha = 0.95)
# plt.plot(t_tab,2.*phiAB,'r--',alpha = 0.95)
plt.plot(t_tab,(wAB-2.*phiAB),'r',alpha = 0.95,ms=8)
plt.axhline(y=kamp[Drr_index2],color ='k',alpha = 0.95,ms=8)
plt.ylabel(r'  $\beta \langle \delta(x(0)) \Gamma \rangle_\lambda/2$', {'color': 'k', 'fontsize': 20})
plt.xlabel(r'  $\tau/\tau_A$', {'color': 'k', 'fontsize': 20})
plt.xlim(t_tab[0], t_tab[-1])
plt.ylim(-3,3)
plt.xticks(size=16)
plt.yticks(size=16)


# subplot 2 low
f_ax2_low = fig.add_subplot(gs[0, 1])

# Q(t)
for i in range(maxtraj):
    plt.plot(t_tab,A_inst[i],'b',alpha = 0.025,ms=8)

plt.plot(t_tab,AAB,'r',alpha = 0.95,ms=8)
# plt.plot(t_tab,phiAB,'r--',alpha = 0.95)
# plt.plot(t_tab,(AAB-phiAB),'k--',alpha = 0.95)
plt.axhline(y=kamp[Drr_index2],color ='k',alpha = 0.95,ms=8)
plt.ylabel(r'  $\beta \langle \delta(x(0)) Q \rangle_\lambda/2$', {'color': 'k', 'fontsize': 20})
plt.xlabel(r'  $\tau/\tau_A$', {'color': 'k', 'fontsize': 20})
plt.xlim(t_tab[0], t_tab[-1])
plt.ylim(-3,3)
plt.xticks(size=16)
plt.yticks(size=16)


# subplot 3 up
f_ax3_up = fig.add_subplot(gs[1, 2])
Drr,t_obs = Drr_tab[Drr_index3], t_obs3
print(Drr)
# t_tab, x_inst, w_inst, A_inst,phi_inst,lo, wAB, AAB,phiAB,wA,AA,phiA= get_inst_data(vo,Drr,t_obs, dt)
fname = 'abp_inst_small_vo_'+str(vo)+'_Drr_'+str(Drr)+'_t_obs_'+str(t_obs)+'_dt_'+str(dt)+'.npy'
inst_data_small = np.load(fname, allow_pickle = True)
t_tab, w_inst, A_inst,phi_inst, wAB, AAB,phiAB,wA,AA,phiA = inst_data_small
t_tab = t_tab/t_relax
# print('kamp = ',kamp)
ttm = (t_tab-t_obs)*dt
# A(t)
for i in range(maxtraj):
    plt.plot(t_tab,w_inst[i],'b',alpha = 0.025,ms=8)

# plt.plot(t_tab,wAB,'r',alpha = 0.95)
# plt.plot(t_tab,2.*phiAB,'r--',alpha = 0.95)
plt.plot(t_tab,(wAB-2.*phiAB),'r',alpha = 0.95,ms=8)
plt.axhline(y=kamp[Drr_index3],color ='k',alpha = 0.95,ms=8)
plt.ylabel(r'  $\beta \langle \delta(x(0)) \Gamma \rangle_\lambda/2$', {'color': 'k', 'fontsize': 20})
plt.xlabel(r'  $\tau/\tau_A$', {'color': 'k', 'fontsize': 20})
plt.xlim(t_tab[0], t_tab[-1])
plt.ylim(-3,3)
plt.xticks(size=16)
plt.yticks(size=16)

# subplot 3 low
f_ax3_low = fig.add_subplot(gs[0, 2])

# Q(t)
for i in range(maxtraj):
    plt.plot(t_tab,A_inst[i],'b',alpha = 0.025,ms=8)

plt.plot(t_tab,AAB,'r',alpha = 0.95,ms=8)
# plt.plot(t_tab,phiAB,'r--',alpha = 0.95)
# plt.plot(t_tab,(AAB-phiAB),'k--',alpha = 0.95)
plt.axhline(y=kamp[Drr_index3],color ='k',alpha = 0.95,ms=8)
plt.ylabel(r'  $\beta \langle \delta(x(0)) Q \rangle_\lambda/2$', {'color': 'k', 'fontsize': 20})
plt.xlabel(r'  $\tau/\tau_A$', {'color': 'k', 'fontsize': 20})
plt.xlim(t_tab[0], t_tab[-1])
plt.ylim(-3,3)
plt.xticks(size=16)
plt.yticks(size=16)


plt.show()
