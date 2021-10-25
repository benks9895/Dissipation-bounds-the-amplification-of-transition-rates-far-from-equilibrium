import numpy as np
import math as m
from numba import jit
from multiprocessing import Pool
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy import signal

import random
import datetime

# Last updated 10/07/2020
# instructions for how to use this code will be added at a later date (~10-11/2020)
# until then, please feel free to contact Benjamin Kuznets-Speck (ben_kuznets-speck@berkeley.edu)
# and he will be happy to guide you through its use.


plt.rc('text', usetex=True)

# the piece-wise V(x) = V_A + V_B equilibrium potential
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

# piece-wise equilibrium force
@jit(nopython=True)
def force(x, h1, h2, l12, l22):
    if x<0:

        F = 4.0*(h1*x/l12)*(1.-x**2./l12)
    else:

        F = 4.0*(h2*x/l22)*(1.-x**2./l22)

    return F

# 1st order stochastic Euler integration
@jit(nopython=True)
def integrate(x,theta,t,h1, h2, l12, l22,vo,f, w,Drr, Dtt,dt):
    xn=x+(vo*m.cos(theta)+f*m.cos(w*t))*dt+\
        force(x, h1, h2, l12, l22)*dt+(2*Dtt*dt)**0.5*np.random.randn()
    thetan=theta+(2*Drr*dt)**0.5*np.random.randn()
    t  = t + dt
    return xn, thetan,t

# 1st order stochastic Euler integration, computing heat-rate dQ as well
@jit(nopython=True)
def integrate_w_A(x,theta,t,h1, h2, l12, l22,vo,f, w,Drr, Dtt,dt):
    # time-dependent external force
    ft = vo*m.cos(theta)+f*m.cos(w*t)
    xn=x+ft*dt+\
        force(x, h1, h2, l12, l22)*dt+(2*Dtt*dt)**0.5*np.random.randn()
    thetan=theta+(2*Drr*dt)**0.5*np.random.randn()
    # heat rate
    dQ=0.5*ft*(xn-x)
    t  = t + dt
    # dA=ft*force(x, h1, h2, l1, l2)*dt
    # dP = 0.25*dt*f**2.
    # return xn, thetan, dQ, dA, dP
    return xn, thetan, dQ,t

# h indicator functions, for both states (h) and each state (hA and hB) individually
@jit(nopython=True)
def h(x, xTSA, xTSB):
    if x < xTSA:
        return -1.
    elif x > xTSB:
        return 1.
    else:
        return 0.

@jit(nopython=True)
def hB(x, xTSB):
    if x > xTSB:
        return 1.
    else:
        return 0.

@jit(nopython=True)
def hA(x, xTSA):
    if x < xTSA:
        return 1.
    else:
        return 0.

# Kramers rate
def kramAB( h1, h2, l12, l22):
#     use kramers formaula with same well/barrier frequency to estimate the steps needed for n transitions
    # note diffusion constant has been set to 1
    wA2, wB2 = (8.*h1)/l12, (8.*h2)/l22
    kAB, kBA = (wA2/(2.*m.pi))*m.e**(-h1), (wB2/(2.*m.pi))*m.e**(-h2)
    return kAB, kBA

# computes correlation function <traj1(0) traj2(t)>
# @jit()
def corr(traj1, traj2):
    # traj  = traj - np.mean(traj)
    c0 = signal.correlate(traj1, traj2, mode='full', method='fft')
    # c0 = np.correlate(traj, traj, mode='full') corr.size//2:
    # c0 = c0[np.argmax(c0):]
    c0 = c0[c0.size//2:]/len(traj1)

    # a = np.concatenate((traj,np.zeros(len(traj)-1))) # added zeros to your signal
    # A = np.fft.fft(a)
    # S = np.conj(A)*A
    # c0 = np.fft.ifft(S)
    # c0 = c0[:(c0.size//2)+1]
    return  c0

# estimate of total time-steps needed to see n trasitions from A to B, via Kramers rate
def get_tot_steps(n,dt, h1, h2, l12, l22):
#     use kramers formaula with same well/barrier frequency to estimate the steps needed for n transitions
    wA2, wB2 = (8.*h1)/l12, (8.*h2)/l22
    kram_AB,kram_BA = (wA2/(2.*m.pi))*m.e**(-h1), (wB2/(2.*m.pi))*m.e**(-h2)

    tot_steps = int(round(n*(1./kram_AB + 1./kram_BA)/dt))
    return tot_steps

# generate 1 trajectory, tot_steps long
@jit(nopython=True)
def do_single_traj(param_id, trial_num, tot_steps,h1, h2, l12, l22,vo,f, w,Drr, Dtt,dt, r_seed):
    np.random.seed(r_seed)

    l1, l2 = (l12)**0.5, (l22)**0.5
    rr = np.random.choice(2)
    if rr == True:
        x, theta, t = -l1, 0.0, 0.0
    else:
        x, theta, t = l2, 0.0, 0.0

    t_burn_in = 10**4

    xTSA, xTSB = -(l12/3.)**0.5, (l22/3.)**0.5


    for j in range(t_burn_in):
        x,theta,t = integrate(x,theta,t,h1, h2, l12, l22,vo,f, w,Drr, Dtt,dt)

    x_traj,h_traj,dQ_traj = \
            np.zeros(tot_steps), np.zeros(tot_steps),np.zeros(tot_steps)

    transitions = 0
    max_transitions = 10**4
    # indeces of
    originsAB, originsBA = np.zeros((max_transitions,4)),np.zeros((max_transitions,4))
    # tauA, tauB, tauTSAB, tauTSBA =\
    #         np.zeros(max_transitions), np.zeros(max_transitions), np.zeros(max_transitions),np.zeros(max_transitions)
    countAB, countBA, countTOT = 0,0, 0


    commit_trans = 25
    wA_dot, AA_dot = 0.,0.
    for i in range(tot_steps):

        x,theta, dQi,t = integrate_w_A(x,theta,t,h1, h2, l12, l22,vo,f, w,Drr, Dtt,dt)
        x_traj[i], dQ_traj[i], h_traj[i] = x,dQi, h(x, xTSA, xTSB)

    # now we need to go through hB_traj to find out where the transitions occured
    # namely, we want all the pairs of indeces of x_traj corresponding to transitions A to B and B to A
    # if we get rid of all the 0s in hB_traj, all nearest neighbors
    # if we get rid of all the 0s and -1s we'll have and indicator of when the state is in B
    # A_index, B_index = np.nonzero(h_traj < -.1), np.nonzero(h_traj > .1)

    trans = np.nonzero(np.diff(h_traj**2.))[0]

    lt = len(trans)
    if lt > 0:

        for i in range(1,len(trans)-2):

            # an actual transition occured
            if h_traj[trans[i]] != h_traj[trans[i+2]]:
                # from A to B
                if h_traj[trans[i]] == -1.:
                    originsAB[countAB] = np.array([trans[i-1],trans[i], trans[i+1], trans[i+2]])
                    # tauA[countTOT] = trans[i] - trans[i-1]-1.
                    # tauTSAB[countAB] = trans[i+1] - trans[i]-1.
                    # tauB[countTOT]= trans[i+2] - trans[i+1]-1.
                    countAB += 1
                    countTOT += 1
                    # re
                # from B to A
                else:
                    originsBA[countBA] = np.array([trans[i-1],trans[i], trans[i+1], trans[i+2]])
                    # tauB[countTOT] = trans[i] - trans[i-1]-1.
                    # tauTSBA[countBA] = trans[i+1] - trans[i]-1.
                    # tauA[countTOT]= trans[i+2] - trans[i+1]-1.
                    countBA += 1
                    countTOT += 1

    # tauA = tauA[np.nonzero(tauA)[0]]
    # tauB = tauB[np.nonzero(tauB)[0]]
    # tauTSAB = tauTSAB[np.nonzero(tauTSAB)[0]]
    # tauTSBA = tauTSBA[np.nonzero(tauTSBA)[0]]


    if countAB > 0:
        print('countAB = ',countAB)
        originsAB = originsAB[np.nonzero(originsAB[:,1])[0]]
    else:
        print('no transitions AB')
        originsAB = originsAB[0:1]

    if countBA > 0:
        print('countBA = ',countBA)
        originsBA = originsBA[np.nonzero(originsBA[:,1])[0]]
    else:
        print('no transitions BA')
        originsBA = originsAB[0:1]



    # fname = 'rdw_test_tauA_id'+str(int(param_id))+'_trial'+str(int(trial_num))
    # np.save(fname, tauA)
    # fname = 'rdw_test_tauB_id'+str(int(param_id))+'_trial'+str(int(trial_num))
    # np.save(fname, tauB)
    # fname = 'rdw_test_tauTSAB_id'+str(int(param_id))+'_trial'+str(int(trial_num))
    # np.save(fname, tauTSAB)
    # fname = 'rdw_test_tauTSBA_id'+str(int(param_id))+'_trial'+str(int(trial_num))
    # np.save(fname, tauTSBA)
    # np.save(outfile, x)
    # np.save(outfile, x)
    # print('countAB, countBA, countTOT = ',countAB, countBA, countTOT)
    # return x_traj, dQ_traj, h_traj,originsAB, originsBA, tauA, tauB, tauTSAB, tauTSBA
    return x_traj,dQ_traj,h_traj,originsAB, originsBA, countAB, countBA
# generate n trajectories, tot_steps long
def get_n_trajs(header,num_trajs, param_id, tot_steps,h1, h2, l12, l22,vo,f, w,Drr, Dtt,dt, save_traj):
    r_seed = np.random.choice(10**4)
    trial_num = 0
    countAB, countBA = 0,0
    while True:
        r_seed = np.random.choice(10**4)

        x_traj,dQ_traj,h_traj,originsAB, originsBA, countABi, countBAi =\
         do_single_traj(param_id, trial_num, tot_steps,h1, h2, l12, l22,vo,f, w,Drr, Dtt,dt, r_seed)



        if countABi > 0 and  countBAi > 0 :
            if save_traj:
                print('saving started')
                fname = header+'_x_traj_id_'+str(int(param_id))+'_trial_'+str(int(trial_num))
                np.save(fname, x_traj)
                fname = header+'_dQ_traj_id_'+str(int(param_id))+'_trial_'+str(int(trial_num))
                np.save(fname, dQ_traj)
                fname = header+'_h_traj_id_'+str(int(param_id))+'_trial_'+str(int(trial_num))
                np.save(fname, h_traj)
                fname = header+'_originsAB_id_'+str(int(param_id))+'_trial_'+str(int(trial_num))
                np.save(fname, originsAB)
                fname = header+'_originsBA_id_'+str(int(param_id))+'_trial_'+str(int(trial_num))
                np.save(fname, originsBA)
            else:
                print('saving started')
                fname = header+'_x_traj_id_'+str(int(0))+'_trial_'+str(int(trial_num))
                np.save(fname, x_traj)
                fname = header+'_dQ_traj_id_'+str(int(0))+'_trial_'+str(int(trial_num))
                np.save(fname, dQ_traj)
                fname = header+'_h_traj_id_'+str(int(0))+'_trial_'+str(int(trial_num))
                np.save(fname, h_traj)
                fname = header+'_originsAB_id_'+str(int(0))+'_trial_'+str(int(trial_num))
                np.save(fname, originsAB)
                fname = header+'_originsBA_id_'+str(int(0))+'_trial_'+str(int(trial_num))
                np.save(fname, originsBA)



            countAB += countABi
            countBA += countBAi
            trial_num += 1
            print('saving finished')

        if countAB+ countBA > 2.*num_trajs:
            # print('saving finished')
            break

    rdw_archive = np.array([countAB,trial_num])
    print('countAB,trial_num=',countAB,trial_num+1)
    fname = header+'_archive_id_'+str(param_id)

    np.save(fname,rdw_archive)
# analyses all data from param_id to find the observation time needed
# to let 99% of transition paths to complete
def find_tobs( header,param_id, save_traj):

    fname = header+'_archive_id_'+str(param_id)+'.npy'
    countAB, trial_num = np.load(fname)
    cAB = 0
    tauAB, tauTS = [], []
    for i in range(trial_num):
        # print(i)
        if save_traj:
            fname = header+'_originsAB_id_'+str(int(param_id))+'_trial_'+str(int(i))+'.npy'
        else:
            fname = header+'_originsAB_id_'+str(int(0))+'_trial_'+str(int(i))+'.npy'
        originsAB = np.load(fname)
        for j in range(len(originsAB)):
            # if originsAB >
            tauAB.append(originsAB[j][3]-originsAB[j][0])
            tauTS.append(originsAB[j][2]-originsAB[j][1])

    tauAB = np.asarray(tauAB)
    tauTS = np.asarray(tauTS)
    # print(np.mean(tauAB), np.std(tauAB))
    # print(np.mean(tauTS), np.std(tauTS))

    n_bins = 20
    tAB_hist_tot, tTS_hist_tot = np.histogram(tauAB, n_bins), np.histogram(tauTS, n_bins)

    dbAB, dbTS = tAB_hist_tot[1][-1]-tAB_hist_tot[1][-2],tTS_hist_tot[1][-1]-tTS_hist_tot[1][-2]
    tAB_bins_tot, tTS_bins_tot = tAB_hist_tot[1], tTS_hist_tot[1]
    tAB_bins, tTS_bins = tAB_bins_tot[:-1]+dbAB/2., tTS_bins_tot[:-1]+dbTS/2.

    t_thresh = 1.-10**-2.
    tc = 0.
    ntot = np.sum(tTS_hist_tot[0])
    ltt = len(tTS_hist_tot[0])
    tobs = 0
    for i in range(ltt-1):
        tc += tTS_hist_tot[0][i]/ntot
        if tc>t_thresh:
            tobs = int(tTS_bins[i])
            break
    if tobs == 0:
        tobs = int(max(tTS_bins))

    tobs = int(1.5*tobs)
    # tobs = int(round(np.mean(tauAB)))
    # tobs = int(2*round(np.mean(tauTS)))
    print('tobs = ',tobs)

    # tTS_hist = -np.log(tTS_hist_tot[0])-np.min(-np.log(tTS_bins_tot[0]))
    # tAB_hist = -np.log(tAB_hist_tot[0])-np.min(-np.log(tAB_hist_tot[0]))
    # tTS_hist = tTS_hist_tot[0]
    # tAB_hist = tAB_hist_tot[0]
    #
    # tbar = int(round(np.mean(tauTS)))
    # plt.figure()
    # plt.plot(tTS_bins,tTS_hist,'o')
    # plt.axvline(x=tobs,alpha = 0.5)
    # plt.axvline(x=np.mean(tauTS),alpha = 0.5, color = 'r')
    # # np.mean(tauTS)
    # plt.show()
    #
    # plt.figure()
    # plt.plot(tAB_bins,tAB_hist,'o')
    # plt.show()

    return tobs


# computes  <hB(t)hA(0)> given hA(t) and hB(t)
def cAB_traj(hAt, hBt, tobs):

    hA, hB = np.mean(hAt),np.mean(hBt)
    hAB = corr(hAt,hBt)

    return hAB[:tobs],hA, hB


# compute the average heat Q integrated over trajectories of length tobs that contain a transition
@jit(nopython=True)
def compute_Q_single( tobs,n_coarse,  n_dto,  originsAB, dQ_traj, h_traj):

    lq, lo = len(dQ_traj), len(originsAB)
    dto = int(m.floor(tobs/n_dto))
    Q_inst_now = np.zeros(n_dto)
    Q_inst = np.zeros(n_dto)
    QAB = 0.


    ct = 0.
    # compute Q
    for i in range(lo):

    # we want to move a window of len tobs
        # from the point in dQ where the transition first occurs (0 -> 1) = origins 2
        # print('tobs = ',tobs)
        for j in np.arange(0,tobs, n_coarse):

            tf0 = int(originsAB[i][2])
            # make sure whole traj of len tobs falls in the traj dQ
            tf, t0 = tf0 + n_coarse*j, tf0-tobs+n_coarse*j
            if t0 > 0. and tf < lq:
                # and check to see if the transition happened
                # print(t0)
                if h_traj[t0] == -1. and h_traj[tf] == 1.:
                    QAB  = QAB + np.sum(dQ_traj[t0:t0+ tobs])
                    for k in range(n_dto):
                        Q_inst_now[k] = np.sum(dQ_traj[t0:t0+dto*k+1])
                    Q_inst = Q_inst + Q_inst_now
                    ct += 1.
    print('ct = ',ct)

    if ct > 0:
        QAB = QAB/ct
        Q_inst = Q_inst/ct

    return ct, QAB, Q_inst



def compute_cAB_single( tobs, h_traj):

    hA_traj, hB_traj = h_traj, h_traj
    hA_traj[np.nonzero(hA_traj>0.)[0]][:] = 0.
    hA_traj = -1.*hA_traj
    hB_traj[np.nonzero(hB_traj<0.)[0]][:] = 0.

    hAB,hA, hB = cAB_traj(hA_traj, hB_traj, tobs)

    return hAB,hA, hB


def compute_cAB_Q_n(getcAB, getQ,header,param_id, n_coarse,n_dto,save_traj):

    tobs = find_tobs(header, param_id,save_traj)
    fname = header+'_tobs_'+str(param_id)
    np.save(fname, tobs)
    nto = 1.

    fname = header+'_archive_id_'+str(param_id)+'.npy'
    countAB, trial_num = np.load(fname)

    if getQ:
        Q_inst = np.zeros((trial_num, n_dto))
        QAB  = np.zeros(trial_num)
        ct = np.zeros(trial_num)


    # we want to make sure that tobs is large enough to allow at least one transition
    # for the chosen coarse grained converge_time
    # if no transitions occur, this means we're skipping over them and need to have a larger tobs
    start_again = False

    while True:
        start_again = False
        if save_traj:
            fname = header+'_dQ_traj_id_'+str(int(param_id))+'_trial_'+str(int(0))+'.npy'
            dQ_traj = np.load(fname)
            fname = header+'_h_traj_id_'+str(int(param_id))+'_trial_'+str(int(0))+'.npy'
            h_traj = np.load(fname)
            fname = header+'_originsAB_id_'+str(int(param_id))+'_trial_'+str(int(0))+'.npy'
            originsAB = np.load(fname)
        else:
            fname = header+'_dQ_traj_id_'+str(int(0))+'_trial_'+str(int(0))+'.npy'
            dQ_traj = np.load(fname)
            fname = header+'_h_traj_id_'+str(int(0))+'_trial_'+str(int(0))+'.npy'
            h_traj = np.load(fname)
            fname = header+'_originsAB_id_'+str(int(0))+'_trial_'+str(int(0))+'.npy'
            originsAB = np.load(fname)

        if getQ:
            ct[0], QAB[0], Q_inst[0] = compute_Q_single( tobs,n_coarse,n_dto, originsAB, dQ_traj, h_traj)
        if getcAB:
            hAB ,hA, hB = compute_cAB_single( tobs, h_traj)

        for i in range(1,trial_num):
            print('dq frac = ',i/trial_num)

            if save_traj:

                fname = header+'_dQ_traj_id_'+str(int(param_id))+'_trial_'+str(int(i))+'.npy'
                dQ_traj = np.load(fname)
                fname = header+'_h_traj_id_'+str(int(param_id))+'_trial_'+str(int(i))+'.npy'
                h_traj = np.load(fname)
                fname = header+'_originsAB_id_'+str(int(param_id))+'_trial_'+str(int(i))+'.npy'
                originsAB = np.load(fname)
            else:
                fname = header+'_dQ_traj_id_'+str(int(0))+'_trial_'+str(int(i))+'.npy'
                dQ_traj = np.load(fname)
                fname = header+'_h_traj_id_'+str(int(0))+'_trial_'+str(int(i))+'.npy'
                h_traj = np.load(fname)
                fname = header+'_originsAB_id_'+str(int(0))+'_trial_'+str(int(i))+'.npy'
                originsAB = np.load(fname)

            if getQ:
                ct[i], QAB[i], Q_inst[i] = compute_Q_single( tobs,n_coarse,n_dto, originsAB, dQ_traj, h_traj)
                if ct[i] == 0.:
                    start_again = True
                    nto += 0.25
                    tobs = int(round(nto*tobs))
                    break
            if getcAB:
                hABi,hAi, hBi = compute_cAB_single( tobs, h_traj)
                hAB = hAB + hABi
                hA = hA + hAi
                hB = hB + hBi


        if start_again == False:
            break
        else:
            print('starting again with tobs =', tobs )

        # ct[i] = cti
        # QAB[i] = QABi
        # print(Q_insti)
        # Q_inst[i] = Q_insti

    if getcAB:
        cAB = hAB/hA
        fname = header+'_cAB_id_'+str(int(param_id))
        np.save(fname, cAB)

    if getQ:
        mQAB, mQAB_err = np.mean(QAB), np.std(QAB)/trial_num**0.5
        mQ_inst, mQ_inst_err = np.mean(Q_inst, axis = 0), np.std(Q_inst, axis = 0)/trial_num**0.5
        fname = header+'_QAB_id_'+str(int(param_id))
        np.save(fname, mQAB)
        fname = header+'_QAB_err_id_'+str(int(param_id))
        np.save(fname, mQAB_err)
        fname = header+'_Q_inst_id_'+str(int(param_id))
        np.save(fname, mQ_inst)
        fname = header+'_Q_inst_err_id_'+str(int(param_id))
        np.save(fname, mQ_inst_err)
        print('finished')

# creates a parameter database with specified header which fixes the equilibrium potential V(x) and the driving force
def draw_params(header, n, h1_min, h1_max, h2_min, h2_max, wA_min, wA_max, wB_min, wB_max):

    # params will be h1, h2, l1, l2, f, vo, Drr, omega
    Dtt = 1.
    params = np.zeros((n,9))
    h1_tab = np.random.uniform(h1_min,h1_max,n)
    h2_tab = np.random.uniform(h2_min,h2_max,n)
    # this will set a maximum and minimum curvature in each well
    l12_min_tab = 8.*h1_tab/wA_max**2.
    l12_max_tab = 8.*h1_tab/wA_min**2.
    l22_min_tab = 8.*h2_tab/wB_max**2.
    l22_max_tab = 8.*h2_tab/wB_min**2.


    for i in range(n):
        l12i = np.random.uniform(l12_min_tab[i],l12_max_tab[i])
        l22i = np.random.uniform(l22_min_tab[i],l22_max_tab[i])
        # we want |f|+|v| < maxforce =
        # 1st pick fraction from 0 to 1
        frac_f = np.random.uniform()
        # next draw a number from 1 to maxforce

        draw_force = np.random.uniform(2.,(h1_tab[i])/2.)
        fi = frac_f*draw_force
        voi = (1.-frac_f)*draw_force
        wi = np.random.uniform(min(wA_min,wB_min), max(wA_max,wB_max))
        Drri = np.random.uniform(Dtt/10., Dtt*10.)
        # print(np.array([h1_tab[i], h2_tab[i],l12i, l22i, voi, fi,wi, Drri, Dtt]))
        params[i] = np.around(np.array([h1_tab[i], h2_tab[i],l12i, l22i, voi, fi,wi, Drri, Dtt])\
                    , decimals=1)

    fname = header+'_param_archive'
    np.save(fname, params)
    print('param archive created')

# sample parameters below
# n = 10**2
# h1_min, h1_max = 3.,7.
# h2_min, h2_max = 3.,7.
# wA_min, wA_max = 4., 7.5
# wB_min, wB_max = 4., 7.5
# # header = 'rdw_cABQ'
# dt = 0.001

def Q_plot(header,param_id, dt):


    n_header = header +'_neq'
    e_header = header +'_eq'
    fname = header+'_param_archive.npy'
    params = np.load(fname)
    h1, h2, l12, l22, vo,f,w,Drr,Dtt = params[param_id]
    print('h1, h2, l12, l22, vo,f,w,Drr,Dtt  = ',h1, h2, l12, l22, vo,f,w,Drr,Dtt )
    t_relax = 1/(8.*h1/l12)**0.5

    # plt.figure()
    # xxp, pxx = potential(header, param_id )
    # plt.plot(xxp, pxx)
    # plt.show()

    fname = n_header+'_tobs_'+str(param_id)+'.npy'
    tobs_n = np.load(fname)
    fname = e_header+'_tobs_'+str(param_id)+'.npy'
    tobs_e = np.load(fname)

    tobs = min(tobs_n,tobs_e)

    fname = n_header+'_QAB_id_'+str(int(param_id))+'.npy'
    QAB = np.load(fname)
    fname = n_header+'_QAB_err_id_'+str(int(param_id))+'.npy'
    QAB_err = np.load(fname)
    fname = n_header+'_Q_inst_id_'+str(int(param_id))+'.npy'
    Q_inst = np.load(fname)
    fname = n_header+'_Q_inst_err_id_'+str(int(param_id))+'.npy'
    Q_inst_err = np.load(fname)



    fname = n_header+'_cAB_id_'+str(int(param_id))+'.npy'
    cABn = np.load(fname)
    fname = e_header+'_cAB_id_'+str(int(param_id))+'.npy'
    cABe = np.load(fname)



    n_dto = 50
    dto = int(m.floor(tobs/n_dto))
    tt  = int(len(Q_inst)*n_dto)

    ttC = np.arange(tobs)*dt/t_relax
    ttQ = np.arange(len(Q_inst))*dt*dto/t_relax


    kamp = np.log(np.diff(cABn[:tobs])/np.diff(cABe[:tobs]))

    # le, ln = len(cABe),len(cABn)
    # if le<ln:
    #     ttfull = np.arange(le)*dt
    #     kamp = np.log(np.diff(cABn[:le])/np.diff(cABe[:le]))
    # else:
    #     ttfull = np.arange(ln)*dt
    #     kamp = np.log(np.diff(cABn[:ln])/np.diff(cABe[:ln]))
    plt.figure()
    # plt.plot(ttQ[1:], np.diff(Q_inst)/(dt*dto), 'bo-',fillstyle='none', alpha = 0.95,ms=8,label=r'  $\beta  \langle \dot Q \rangle_\lambda/2$')
    plt.plot(ttQ, Q_inst, 'ro-',fillstyle='none', alpha = 0.95,ms=8, label=r'  $\beta \langle Q \rangle_\lambda/2$')

    plt.fill_between(ttQ, Q_inst-Q_inst_err,Q_inst+Q_inst_err,
                 facecolor="grey", # The fill color
                 color='red',       # The outline color
                 alpha=0.3)

    # Qavg = np.zeros(len(Q_inst)-1)
    # for i in range(1,len(Q_inst)):
    #     Qavg[i-1] = dto*dt*np.sum(Q_inst[:i])/ttQ[i]
    # plt.plot(ttQ[1:], Qavg, 'ro-',fillstyle='none', alpha = 0.95,ms=8, label=r'  $\beta \langle Q \rangle_\lambda/2$')
    # plt.plot(tt, Q_inst/tt, 'ro-',fillstyle='none', alpha = 0.95,ms=8)
    plt.plot(ttC[1:], kamp, 'k-',fillstyle='none', alpha = 0.95,ms=8,label=r'  $\ln \dot{C_\lambda}/\dot{C_0}$')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.,fontsize=20,frameon = False)
    # plt.axhline(y=kamp,color ='k', linestyle = '-')
    # plt.axhline(y=kamp+kamp_err,color ='k', linestyle = '-')
    # plt.axhline(y=kamp-kamp_err,color ='k', linestyle = '-')
    plt.xticks(size=16)
    plt.yticks(size=16)

    # plt.ylabel(r'  $x(t)/l_A$', {'color': 'k', 'fontsize': 20})
    plt.xlabel(r'  $t/\tau_A $', {'color': 'k', 'fontsize': 20})
    # plt.xlabel(r'  $time$', {'color': 'k', 'fontsize': 20, 'fontweight':'bold'})
    plt.tight_layout()
    f_name = header+'_plot_'+str(int(param_id))+'.pdf'
    plt.savefig(f_name)
    plt.show()

# Q_plot(header,param_id, dt)

def do_cAB_Q_routine(dt,n_coarse,n_dto,save_traj,need_trajs, header, param_id,getcAB, getQ, plot_cAB_Q, plot_pot ):

    n_header = header +'_neq'
    e_header = header +'_eq'


    fname = header+'_param_archive.npy'
    params = np.load(fname)
    h1, h2, l12, l22, vo,f,w,Drr,Dtt = params[param_id]
    print('h1, h2, l12, l22, vo,f,w,Drr,Dtt  = ',h1, h2, l12, l22, vo,f,w,Drr,Dtt )

    if plot_pot:
        plt.figure()
        xxp, pxx = potential(header, param_id )
        plt.plot(xxp, pxx)
        plt.show()

    num_trajs = 2500
    n = 100
    tot_steps = get_tot_steps(n,dt, h1, h2, l12, l22)

    if need_trajs:
        print('getting n = ',str(num_trajs), ' neq reactions')
        get_n_trajs(n_header,num_trajs, param_id, tot_steps,h1, h2, l12, l22,vo,f, w,Drr, Dtt,dt,save_traj)
        print('getting n = ',str(num_trajs), ' eq reactions')
        get_n_trajs(e_header,num_trajs, param_id, tot_steps,h1, h2, l12, l22,0.,0., 0.,Drr, Dtt,dt,save_traj)


    print('computing neq heat and rate')
    compute_cAB_Q_n(getcAB, getQ,n_header,param_id, n_coarse,n_dto,save_traj)
    if getcAB:
        print('computing eq heat and rate')
        compute_cAB_Q_n(getcAB, getQ,e_header,param_id, n_coarse,n_dto,save_traj)
    if plot_cAB_Q == True:

        Q_plot(header,param_id, dt)

param_id = 55
# regular
# plot_cAB_Q = True
# getcAB = True
# getQ = True
# n_coarse = 10
# n_dto = 50
# save_traj = False
# need_trajs = True
# plot_pot = True

# adjust Q coarse
plot_cAB_Q = True
getcAB = False
getQ = True
n_coarse =100
n_dto = 50
save_traj = False
need_trajs = False
plot_pot = False

# do_cAB_Q_routine(dt,n_coarse,n_dto,save_traj,need_trajs, header, param_id,getcAB, getQ, plot_cAB_Q,plot_pot)
