import numpy as np
import math as m
from numba import jit
from multiprocessing import Pool
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

plt.rc('text', usetex=True)
fname = 'fig12_tt.npy'
tt = np.load(fname)
fname = 'fig12_x_traj_eq.npy'
x_eq = np.load(fname)
fname = 'fig12_x_traj_neq.npy'
x_neq = np.load(fname)

print('madeit')
fname = 'rdw_data_fin_tot.npy'
h1, h2, l12, l22,vo,f,w,Drr,Dtt,foFmaxA, foFmaxB,\
        wA, wB,p,kamp,kamp_err,Q,Q_err = np.load(fname)
c = 1.5
width, height = c*4.2,c*4.
fig = plt.figure(figsize = (width, height))
plt.tight_layout()

# gs = fig.add_gridspec(2,5)
# ax0 = plt.subplot(gs[0,0:2])
gs = fig.add_gridspec(6,4)
ax0 = plt.subplot(gs[0:2,0:2])

line0, = ax0.plot(tt,x_eq ,'royalblue', alpha = 0.95,linewidth=1.)
plt.rc('text', usetex=True)
plt.fill_between(tt, -1.35,0.,
             facecolor="grey", # The fill color
             color='grey',       # The outline color
             alpha=0.3)
plt.xticks(size=16)
plt.yticks(size=16)
plt.xticks(np.arange(0, 10000, 2000))
# f_ax1.plot(tt,x_eq ,'royalblue', alpha = 0.95,linewidth=1.)
# plt.xlabel(r'  $\mathrm{time}$', {'color': 'k', 'fontsize': 14})
# plt.ylabel(r'  $\mathrm{q~rxn~coord.}$', {'color': 'k', 'fontsize': 20})
plt.ylabel(r'  $x(t)/l_A$', {'color': 'k', 'fontsize': 20})
plt.xlabel(r'  $t D_x/ l_A^2$', {'color': 'k', 'fontsize': 20})
plt.xlim([0,10000])
plt.ylim([-1.45,1.45])
# plt.fill_between(tt, -1.35,-0.35,
#              facecolor="grey", # The fill color
#              color='grey',       # The outline color
#              alpha=0.3)
# plt.ylim([10**-2,10])
# f_ax1.set_title('gs[0, :]')
# ax1 = plt.subplot(gs[1, 0:2], sharex = ax0)
ax1 = plt.subplot(gs[0:2, 2:], sharex = ax0)
line1, = ax1.plot(tt,x_neq ,'red', alpha = 0.9,linewidth=1.)
# plt.setp(ax0.get_xticklabels(), visible=False)
plt.setp(ax1.get_yticklabels(), visible=False)
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
# plt.subplots_adjust(hspace=.0)
# plt.subplots_adjust(wspace = 1.2)
plt.subplots_adjust(hspace=2.)
plt.subplots_adjust(wspace = 0.)

# f_ax2 = fig.add_subplot(gs[1, 0], sharex = f_ax1)
# f_ax2.plot(tt,x_neq ,'red', alpha = 0.9,linewidth=1.)
plt.xlim([0,10000])
plt.ylim([-1.45,1.45])
plt.rc('text', usetex=True)
plt.xlabel(r'  $t D_x/ l_A^2$', {'color': 'k', 'fontsize': 20})
# plt.setp(f_ax2.get_xticklabels(), visible=False)
# plt.ylabel(r'  $\mathrm{q~rxn~coord.}$', {'color': 'k', 'fontsize': 20})
plt.fill_between(tt, -1.35,0.,
             facecolor="grey", # The fill color
             color='grey',       # The outline color
             alpha=0.3)


plt.xticks(size=16)
plt.yticks(size=16)
plt.xticks(np.arange(0, 10000, 2000))


# plt.subplots_adjust(hspace=.0)
# f_ax2.set_title('gs[1, :-1]')
# f_ax3 = fig.add_subplot(gs[:, 2:5])
f_ax3 = fig.add_subplot(gs[2:, :])
# f_ax3.set_title('gs[1:, -1]')

# area, colors = 30*np.ones(len(p)), np.log(p/(1-p))

# area, colors = 30*np.ones(len(p)), (f+vo)
Da = (f+vo)
area, colors = 30*np.ones(len(p)), (np.log(Da)-np.min(np.log(Da)))/np.max((np.log(Da)-np.min(np.log(Da))))

# area, colors = 30*np.ones(len(p)), Da/np.max(Da)
# area, colors = 30*np.ones(len(p)), 1.01+np.log(Da/(1.+Da))/np.amax(np.absolute(np.log(Da/(1.+Da))))
# area, colors = 30*np.ones(len(p)), np.log(Da)/abs(np.amax(np.log(Da)))
# area, colors = 30*np.ones(len(p)), np.log(foFmaxA)/np.amax(np.log(foFmaxA))

# area, colors = 30*np.ones(len(p)),1/w**2.

# area, colors = 20*np.abs(Drr/(wA*l12)), -(f+vo)
# area, colors = wA/wB,foFmaxA
# np.abs(h1*l12**0.5-h2*l22**0.5)
# np.abs(Drr/(wA*l12))
# cmap = 'Spectral'
# cmap = 'RdBu'
xx,yy = np.linspace(min(Q), max(Q),200),np.linspace(min(kamp), max(kamp),200)
plt.plot(Q,Q,'k-', linewidth=3.75)
plt.scatter( Q,kamp, s=area, c=colors, cmap = 'coolwarm', alpha=0.45)

plt.xlim([0.01,20])
plt.ylim([10**-2,6.5])

plt.xticks(size=16)
plt.yticks(size=16)
plt.yscale('log')
plt.xscale('log')



# plt.xlabel(r'  $\mathrm{dissipation}~=~\beta \langle Q \rangle_\mathrm{AB}/2$', {'color': 'k', 'fontsize': 20})
# plt.ylabel(r'  $\mathrm{rate~enhancement}~=~\ln k^\mathrm{neq}_\mathrm{AB}/k^\mathrm{eq}_\mathrm{AB}$', {'color': 'k', 'fontsize': 20})
plt.xlabel(r'  $\beta \langle Q \rangle_\lambda/2$', {'color': 'k', 'fontsize': 20})
plt.ylabel(r'  $\ln k_\lambda/k_0$',\
        {'color': 'k','fontsize': 20})

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# cbaxes = inset_axes(f_ax3, width="30%", height="4%",  loc= 10)
cbaxes = f_ax3.inset_axes( [0.65, 0.15, 0.3, 0.03])
cbar = plt.colorbar(cax=cbaxes, ticks = [0.,1],orientation='horizontal')
cbaxes.tick_params(labelsize=16)
# cbar.ax.tick_params(labelsize=12)
print('madeit')

# plt.ylabel(  'dissipation', {'color': 'k', 'fontsize': 20, 'fontweight':'bold'})
# plt.xlabel(  'rate enhancement', {'color': 'k', 'fontsize': 20, 'fontweight':'bold'})
# plt.tight_layout()
# f_name = 'rdw_Qt_plot_'+str(int(param_id))+'.pdf'
# plt.savefig(f_name)
plt.tight_layout()
plt.show()
