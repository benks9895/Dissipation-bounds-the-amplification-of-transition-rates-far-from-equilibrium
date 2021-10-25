import numpy as np
import math as m
from numba import jit
from multiprocessing import Pool
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

def potential(header,param_id):

    fname = header+'_param_archive.npy'
    params = np.load(fname)
    h1, h2, l12, l22, vo,f,w,Drr,Dtt = params[param_id]
    xxp = np.linspace(-1.5*l12**0.5, 1.5*l22**0.5,100)
    pxx = np.zeros(len(xxp))
    for i in range(len(xxp)):
        if xxp[i]<0:
            x21 = xxp[i]**2./(2.*l12)
            pxx[i] = 4.*h1*x21*(x21-1.)
        else:
            x22 = xxp[i]**2./(2.*l22)
            pxx[i] = 4.*h2*x22*(x22-1.)
    return xxp, pxx


plt.rc('text', usetex=True)



dt = 0.001

# header1, header2, header3 = 'rdw_cABQ', 'rdw_cABQ_2', 'rdw_cABQ_3'
# param_id1, param_id2, param_id3 = 2,4,4

# 'rdw_cABQ_5','rdw_cABQ_5'
# 9,11
header1, header2, header3 = 'rdw_cABQ', 'rdw_cABQ_5', 'rdw_cABQ'
param_id1, param_id2, param_id3 = 2, 38,5

headers = np.array([header1, header2, header3])
param_ids = np.array([param_id1, param_id2, param_id3])


fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(1,3)


# subplot 1
f_ax1 = fig.add_subplot(gs[0, 0])
header, param_id = header1, param_id1
n_header = header +'_neq'
e_header = header +'_eq'
fname = header+'_param_archive.npy'
print(fname)
params = np.load(fname)
h1, h2, l12, l22, vo,f,w,Drr,Dtt = params[param_id]
print('h1, h2, l12, l22, vo,f,w,Drr,Dtt  = ',h1, h2, l12, l22, vo,f,w,Drr,Dtt )
t_relax = 1/(8.*h1/l12)**0.5
# plt.figure()
# xxp, pxx = potential(header, param_id )
# plt.plot(xxp, pxx)
# plt.show()
fname = n_header+'_tobs_'+str(param_id)+'.npy'
print(fname)
tobs_n = np.load(fname)
fname = e_header+'_tobs_'+str(param_id)+'.npy'
print(fname)
tobs_e = np.load(fname)
tobs = min(tobs_n,tobs_e)
fname = n_header+'_QAB_id_'+str(int(param_id))+'.npy'
print(fname)
QAB = np.load(fname)
fname = n_header+'_QAB_err_id_'+str(int(param_id))+'.npy'
print(fname)
QAB_err = np.load(fname)
fname = n_header+'_Q_inst_id_'+str(int(param_id))+'.npy'
print(fname)
Q_inst = np.load(fname)
fname = n_header+'_Q_inst_err_id_'+str(int(param_id))+'.npy'
print(fname)
Q_inst_err = np.load(fname)
fname = n_header+'_cAB_id_'+str(int(param_id))+'.npy'
print(fname)
cABn = np.load(fname)
fname = e_header+'_cAB_id_'+str(int(param_id))+'.npy'
print(fname)
cABe = np.load(fname)
n_dto = 50
dto = int(m.floor(tobs/n_dto))
tt  = int(len(Q_inst)*n_dto)
ttC = np.arange(tobs)*dt/t_relax
ttQ = np.arange(len(Q_inst))*dt*dto/t_relax
kamp = np.log(np.diff(cABn[:tobs])/np.diff(cABe[:tobs]))
plt.plot(ttQ, Q_inst, 'ro-',fillstyle='none', alpha = 0.95,ms=8, label=r'  $\beta \langle Q \rangle_\lambda/2$')
plt.fill_between(ttQ, Q_inst-Q_inst_err,Q_inst+Q_inst_err,
             facecolor="grey", # The fill color
             color='red',       # The outline color
             alpha=0.3)
plt.plot(ttC[1:], kamp, 'k-',fillstyle='none', alpha = 0.95,ms=8,label=r'  $\ln k_\lambda/k_0$')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.,fontsize=20,frameon = False)
# plt.axhline(y=kamp,color ='k', linestyle = '-')
# plt.axhline(y=kamp+kamp_err,color ='k', linestyle = '-')
# plt.axhline(y=kamp-kamp_err,color ='k', linestyle = '-')
plt.xticks(size=16)
plt.yticks(size=16)
plt.xlabel(r'  $\tau/\tau_A $', {'color': 'k', 'fontsize': 20})



f_ax2 = fig.add_subplot(gs[0, 1])
header, param_id = header2, param_id2
n_header = header +'_neq'
e_header = header +'_eq'
fname = header+'_param_archive.npy'
print(fname)
params = np.load(fname)
h1, h2, l12, l22, vo,f,w,Drr,Dtt = params[param_id]
print('h1, h2, l12, l22, vo,f,w,Drr,Dtt  = ',h1, h2, l12, l22, vo,f,w,Drr,Dtt )
t_relax = 1/(8.*h1/l12)**0.5
# plt.figure()
# xxp, pxx = potential(header, param_id )
# plt.plot(xxp, pxx)
# plt.show()
fname = n_header+'_tobs_'+str(param_id)+'.npy'
print(fname)
tobs_n = np.load(fname)
fname = e_header+'_tobs_'+str(param_id)+'.npy'
print(fname)
tobs_e = np.load(fname)
tobs = min(tobs_n,tobs_e)
fname = n_header+'_QAB_id_'+str(int(param_id))+'.npy'
print(fname)
QAB = np.load(fname)
fname = n_header+'_QAB_err_id_'+str(int(param_id))+'.npy'
print(fname)
QAB_err = np.load(fname)
fname = n_header+'_Q_inst_id_'+str(int(param_id))+'.npy'
print(fname)
Q_inst = np.load(fname)
fname = n_header+'_Q_inst_err_id_'+str(int(param_id))+'.npy'
print(fname)
Q_inst_err = np.load(fname)
fname = n_header+'_cAB_id_'+str(int(param_id))+'.npy'
print(fname)
cABn = np.load(fname)
fname = e_header+'_cAB_id_'+str(int(param_id))+'.npy'
print(fname)
cABe = np.load(fname)
n_dto = 50
dto = int(m.floor(tobs/n_dto))
tt  = int(len(Q_inst)*n_dto)
ttC = np.arange(tobs)*dt/t_relax
ttQ = np.arange(len(Q_inst))*dt*dto/t_relax
kamp = np.log(np.diff(cABn[:tobs])/np.diff(cABe[:tobs]))
plt.plot(ttQ, Q_inst, 'ro-',fillstyle='none', alpha = 0.95,ms=8, label=r'  $\beta \langle Q \rangle_\lambda/2$')
plt.fill_between(ttQ, Q_inst-Q_inst_err,Q_inst+Q_inst_err,
             facecolor="grey", # The fill color
             color='red',       # The outline color
             alpha=0.3)
plt.plot(ttC[1:], kamp, 'k-',fillstyle='none', alpha = 0.95,ms=8,label=r'  $\ln k_\lambda/k_0$')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.,fontsize=20,frameon = False)
# plt.axhline(y=kamp,color ='k', linestyle = '-')
# plt.axhline(y=kamp+kamp_err,color ='k', linestyle = '-')
# plt.axhline(y=kamp-kamp_err,color ='k', linestyle = '-')
plt.xticks(size=16)
plt.yticks(size=16)
plt.xlabel(r'  $\tau/\tau_A $', {'color': 'k', 'fontsize': 20})



f_ax3 = fig.add_subplot(gs[0,2])

header, param_id = header3, param_id3
n_header = header +'_neq'
e_header = header +'_eq'
fname = header+'_param_archive.npy'
print(fname)
params = np.load(fname)
h1, h2, l12, l22, vo,f,w,Drr,Dtt = params[param_id]
print('h1, h2, l12, l22, vo,f,w,Drr,Dtt  = ',h1, h2, l12, l22, vo,f,w,Drr,Dtt )
t_relax = 1/(8.*h1/l12)**0.5
# plt.figure()
# xxp, pxx = potential(header, param_id )
# plt.plot(xxp, pxx)
# plt.show()
fname = n_header+'_tobs_'+str(param_id)+'.npy'
print(fname)
tobs_n = np.load(fname)
fname = e_header+'_tobs_'+str(param_id)+'.npy'
print(fname)
tobs_e = np.load(fname)
tobs = min(tobs_n,tobs_e)
fname = n_header+'_QAB_id_'+str(int(param_id))+'.npy'
print(fname)
QAB = np.load(fname)
fname = n_header+'_QAB_err_id_'+str(int(param_id))+'.npy'
print(fname)
QAB_err = np.load(fname)
fname = n_header+'_Q_inst_id_'+str(int(param_id))+'.npy'
print(fname)
Q_inst = np.load(fname)
fname = n_header+'_Q_inst_err_id_'+str(int(param_id))+'.npy'
print(fname)
Q_inst_err = np.load(fname)
fname = n_header+'_cAB_id_'+str(int(param_id))+'.npy'
print(fname)
cABn = np.load(fname)
fname = e_header+'_cAB_id_'+str(int(param_id))+'.npy'
print(fname)
cABe = np.load(fname)
n_dto = 50
dto = int(m.floor(tobs/n_dto))
tt  = int(len(Q_inst)*n_dto)
ttC = np.arange(tobs)*dt/t_relax
ttQ = np.arange(len(Q_inst))*dt*dto/t_relax
kamp = np.log(np.diff(cABn[:tobs])/np.diff(cABe[:tobs]))
plt.plot(ttQ, Q_inst, 'ro-',fillstyle='none', alpha = 0.95,ms=8, label=r'  $\beta \langle Q \rangle_\lambda/2$')
plt.fill_between(ttQ, Q_inst-Q_inst_err,Q_inst+Q_inst_err,
             facecolor="grey", # The fill color
             color='red',       # The outline color
             alpha=0.3)
plt.plot(ttC[1:], kamp, 'k-',fillstyle='none', alpha = 0.95,ms=8,label=r'  $\ln k_\lambda/k_0$')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.,fontsize=20,frameon = False)
# plt.axhline(y=kamp,color ='k', linestyle = '-')
# plt.axhline(y=kamp+kamp_err,color ='k', linestyle = '-')
# plt.axhline(y=kamp-kamp_err,color ='k', linestyle = '-')
plt.xticks(size=16)
plt.yticks(size=16)
plt.xlabel(r'  $\tau/\tau_A $', {'color': 'k', 'fontsize': 20})


plt.show()

for i in range(len(headers)):
    header = headers[i]
    param_id = param_ids[i]
    plt.figure()
    xxp, pxx = potential(header, param_id )
    plt.plot(xxp, pxx,ms=8)
    plt.show()
