# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:02:34 2021

@author: ernst
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from uncertainties import ufloat
from pycorrelate import pcorrelate, make_loglags

import timeit as timeR
from scipy import linalg
from scipy import sparse
import re

import matplotlib as mpl
new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)

from lmfit import Model

figsize=(4,2.5)
figsize_timetrace=(4,2.5)

def calc_diffusion_coefficient(R=5e-9, eta=1e-3):
    kT = 4.1e-21
    return kT / (6 * np.pi * eta * R)

def calc_box(concentration=1e-12, aspect_ratio=5):
    Avagadro = 6.02e23
    volume_per_molecule = 1e-3 / (concentration * Avagadro)
    box_width = (volume_per_molecule / aspect_ratio) ** (1 / 3)
    box = np.asarray([1, 1, aspect_ratio]) * box_width
    return box

def apply_boundary(position, box):
    for i, x in enumerate(position):
        position[i] = (x + box[i] / 2) % box[i] - box[i] / 2

    return position

def calc_diffusion(N, dt, D_array, p0=[0, 0, 0]):
    L = np.sqrt(2 * D_array * dt)
    L = np.tile(L, (3, 1))

    L_scaling=True
    if L_scaling:
        L = np.sqrt(3)*L #correct for taking 3 steps of L in each coordinate giving a stepsize of sqrt(3*L**2)

    position = np.array([np.random.choice((-1, 1), size=int(N)) for i in range(3)]) * L
    position[:, 0] += p0
    
    
    position = np.cumsum(position, axis=1)
    # position = (position)

    return position

def get_position(n_steps, molecule, dt, D_array, start_positions, box):
    unconstraint_position = calc_diffusion(n_steps, dt, D_array, start_positions[molecule])
    if box is False:
        position=unconstraint_position
    else:
        position = apply_boundary(unconstraint_position, box)
    start_positions[molecule] = position[:, -1]

    return position, start_positions

def convolve_psf(position, I_1, I_2, efficiency=[1.0,1.0], E=0, dir_exc=0, I0=1e4, wavelength=640e-9, NA=1.2, exc_pos=[0, 0, 0],
                 em_pos=[0, 0, 0], aspect_ratio=8, pinhole=50e-6, magnification=60, samples=1):
    
    spot_width = (wavelength / (2 * NA)) * np.array([1, 1, aspect_ratio])

    I = I0 * np.ones_like(position[0])
    for x, xc_exc, xc_em, w in zip(position, exc_pos, em_pos, spot_width):  # excitation
        I *= np.exp(- ((x - xc_exc) ** 2) / (2 * w ** 2)) 
        # The above is true if we take the beam width (w) as the std dev of the Gaussian beam. If we want to take the width where the gaussian beam decays to 1/e**2 then we multiply the exponent by 4.
    I *= ((position[0] - em_pos[0]) ** 2 + (position[1] - em_pos[1]) ** 2) < (
            0.5 * pinhole / magnification) ** 2  # Emission
    I_new = np.repeat(I, samples)
    if np.size(E) != 1:
        E = np.repeat(E, samples)
    # I_1 is the primary intensity. I_2 is another channel which gains intensity 
    # from either FRET or direct excitation from the I_1 excitation wavelength


    #print('\n shapes', samples, np.shape(I_1),np.shape(I_new),np.shape(E), E, '\n')
    I_new_1 = I_new * efficiency[0]
    I_1 += I_new_1 * (1-E)
    I_2 += I_new_1 * E * efficiency[1]
    I_2 += I_new * efficiency[1] * dir_exc
    #print('\n I_1,2', I_1,I_2, '\n')
    return I_1, I_2

def pulses(filename):
    parts = re.split("ns|us|ms|cont|_", filename)
    print('parts:', parts)
    if filename.find('cont') == -1:
        PIE_time = ['ns', 'us', 'ms']
        for time_res in PIE_time:
            if time_res in filename:
                pulse_time__s = int(filename[filename.find('ALEX') + 4:filename.find(time_res)]) \
                                * 10 ** (PIE_time.index(time_res) * 3) / 1e9
                pulse_scheme = [x for x in list(parts[-2]) if x in ['G','R','d']]
                print(pulse_time__s)
                print(pulse_scheme)
    
    #pulse_int = np.array(re.split('R|G|B|P|d', parts[-2].replace('d', 'd0'))[1:], dtype=np.int8)
    #print('p1' ,pulse_int)
    return pulse_time__s, pulse_scheme
    
def apply_PIE(time, I_G514, I_R514, I_G632, I_R632,  
              I_G514_background, I_R514_background, I_G632_background, I_R632_background,
              pulse_time, pulse_scheme, pulse_length):
    
    
    mask_514 = np.nonzero(np.isin(pulse_scheme, 'G'))[0]
    mask_632 = np.nonzero(np.isin(pulse_scheme, 'R'))[0]

    time_mod = (np.round(time / pulse_time, decimals=1) // 1)
    time_mod = time_mod % pulse_length

    bool_514 = np.isin(time_mod, mask_514).astype(int)
    bool_632 = np.isin(time_mod, mask_632).astype(int)
    #print('apply PIE',pulse_scheme, pulse_length, time_mod, mask_514, bool_514 )
    I_G514 += I_G514_background
    I_R514 += I_R514_background
    I_G632 += I_G632_background
    I_R632 += I_R632_background
    I_G514[bool_514 == 0] = 0
    I_R514[bool_514 == 0] = 0
    I_G632[bool_632 == 0] = 0
    I_R632[bool_632 == 0] = 0
    I_G = I_G514 + I_G632
    I_R = I_R514 + I_R632

    return I_G, I_R

def calc_timetags(time, I, sample_freq=1e7):
    recorded = np.random.random(len(time)) < I / sample_freq
    timetags = time[recorded == 1]

    return timetags

def fill_timetags(time, timetags, last_timetag, I, I_background, sample_freq=1e7):
    new_timetags = calc_timetags(time, I + I_background, sample_freq=sample_freq)
    #timetags[last_timetag:last_timetag + len(new_timetags)] = new_timetags
    timetags=np.append(timetags, new_timetags)
    last_timetag += len(new_timetags)

    return timetags, last_timetag

def select_timetags(timetags_G, timetags_R, pulse_time, pulse_scheme, pulse_length):
    mask_514 = np.nonzero(np.isin(pulse_scheme, 'G'))[0]
    mask_632 = np.nonzero(np.isin(pulse_scheme, 'R'))[0]

    timetagsG_mod = (np.round(timetags_G / pulse_time, decimals=1) // 1)
    timetagsG_mod = timetagsG_mod % pulse_length
    timetagsR_mod = (np.round(timetags_R / pulse_time, decimals=1) // 1)
    timetagsR_mod = timetagsR_mod % pulse_length
    
    bool_G514 = np.isin(timetagsG_mod, mask_514).astype(int)
    bool_R514 = np.isin(timetagsR_mod, mask_514).astype(int)
    bool_G632 = np.isin(timetagsG_mod, mask_632).astype(int)
    bool_R632 = np.isin(timetagsR_mod, mask_632).astype(int)

    timetags_G514 = timetags_G[bool_G514 == 1]
    timetags_R514 = timetags_R[bool_R514 == 1]
    timetags_G632 = timetags_G[bool_G632 == 1]
    timetags_R632 = timetags_R[bool_R632 == 1]

    return timetags_G514, timetags_R514, timetags_G632, timetags_R632


def assign_FRET_coefficient(states, E_states, D):
    E = np.zeros_like(states).astype(float)
    D_array = np.zeros_like(states).astype(float)
    for state, e_state in enumerate(E_states):
        E[states == state] += e_state
        D_array[states == state] += D[state]
    return E, D_array

"""get probability of of multistate system"""

def calc_timestep(matrix, p0=None, t_s=0.1, k_s=1e6, single_molecule=False, show=False):
    """
    Compute a new probability distribution after dt seconds based on matrix of 
    free enegeries, bariers and privious probability distribution.
    Use sparse matrix manipulations to speed up large matrix computations.
    Steps:
    1) delta_g-matrix : delta_g = g_barier - g0
    2) t-matrix : t = ks * exp(- delta_g)
    3) Impose detailed balance: sum of transfer rate rows = 0
    4) p-matrix: p = matrixexponential(t * dt)
    5) new distribution of states: p(t) = T(p-matrix).p0

    For easy introduction see Jones2017MedDescMaking, DOI: 10.1177/0272989X17696997
    """
    if False:  # Convert to sparse g-matrix
        gmatrix = sparse.csc_matrix(gmatrix)

    if False:
        # p0 = np.zeros(len(gmatrix))
        p0[0] = 1

    if single_molecule:
        # pt = np.zeros(len(gmatrix))
        states = np.arange(len(matrix), dtype=int)

        x = np.random.choice(states, p=matrix[p0])
        # pt[x] = 1

        return x
    else:
        return matrix.T.dot(p0)


""" k_s determines the number of attempts made to cross the barrier"""

def get_probability(start_state, dt_E, k_s, gmatrix, n_steps, dt):
    
    if sparse.issparse(gmatrix):
        matrix = gmatrix
        matrix.data -= np.take(matrix.diagonal(), matrix.indices)
        matrix.data = k_s * np.exp(matrix.data)
        matrix -= sparse.dia_matrix((matrix.sum(axis=1).A1,
                                     [0]), np.shape(gmatrix))
        matrix = sparse.linalg.expm(matrix * dt_E)
    else:
        matrix = np.zeros_like(gmatrix)
        for i, g in enumerate(gmatrix):
            matrix[i] = k_s * np.exp(-g + g[i])  # converting energy matrix into transition rate matrix. Check textbook
            matrix[i, i] = 0
            matrix[i, i] = -np.sum(matrix[i])

        matrix = linalg.expm(matrix * dt_E)  # converting rate matrix to probability matrix. Check Jones et al

    if True:  # test use of sparse matrix
        n_time_elements=np.round(n_steps*dt/dt_E, decimals=0).astype(int)
        time = np.arange(0, n_time_elements, 1)*dt_E
        #time = np.arange(0, n_steps*dt, dt_E) # old time, gave error when n_steps*dt became too large.
        counter_0=0
        counter_1=0
        lifetimes_0=[]
        lifetimes_1=[]
        p0 = start_state
        p = [p0]
        for i, dt in enumerate(np.diff(time)):
            p.append(calc_timestep(matrix, p[i], t_s=dt, single_molecule=True))
            if p[-2]==0 and p[-1]==0:
                counter_0+=1
            if p[-2]==0 and p[-1]==1:
                lifetimes_0.append(counter_0)
                counter_0=0
            if p[-2]==1 and p[-1]==1:
                counter_1+=1
            if p[-2]==1 and p[-1]==0:
                lifetimes_1.append(counter_1)
                counter_1=0
        
        return p,lifetimes_0,lifetimes_1


""" simulate diffusion and fluorescence for fluorescence correlation spectroscopy"""


def get_states(states_first, n_molecules, tot_timesteps, dt_E, k_s, gmatrix, lag, n_lags, n_steps, dt, timeratio_kinetics):
    states = np.zeros((n_molecules, n_steps)) 
    states_last = np.zeros(n_molecules, dtype=np.int8)

    lifetimes_0_all=[]
    lifetimes_1_all=[]
    for molecule in tqdm(range(n_molecules), desc=f'State Calculation, lag {lag + 1}/{n_lags}'):
        S, lifetimes_0, lifetimes_1 = get_probability(states_first[molecule], dt_E, k_s, gmatrix, n_steps, dt)
        S = np.repeat(S, timeratio_kinetics)
        states[molecule, :] = S
        states_last[molecule] = states[molecule, -1]   
        
        lifetimes_0_all=np.append(lifetimes_0_all,lifetimes_0)
        lifetimes_1_all=np.append(lifetimes_1_all,lifetimes_1)

    return states, states_last, lifetimes_0_all, lifetimes_1_all

def timetags_to_timetrace(timetags, time):
    dt = np.mean(np.diff(time))
    time = np.append(time, time[-1] + dt)
    timetrace, _ = np.histogram(timetags, time)

    return timetrace

def resample_timetrace(time, I, resampled_time):
    dt = np.mean(np.diff(resampled_time))
    resampled_time = np.append(resampled_time, resampled_time[-1] + dt)
    resampled_I, _ = np.histogram(time, resampled_time, weights=I)

    return resampled_I * np.mean(np.diff(time))

def plot_timetrace(time, timetags, I, I_background, axis, colour_tt, colour_I, label, bintime, xrange, yrange, title, folder, filename):
    plt.figure(figsize=figsize_timetrace)
    t = np.arange(time[0], time[-1], bintime)
    t_shown = t-time[0] #so the plot runs from t=0
    i2_max=0
    
    for timetags_trace, I, I_background,plotaxis,colour_tt,colour_I,label in zip(timetags,I,I_background,axis,colour_tt,colour_I,label):
        i = timetags_to_timetrace(timetags_trace, t)
        plt.step(t_shown, plotaxis*i, color=colour_tt, label=label+' + shotnoise') 
        i2 = resample_timetrace(time, I + I_background, t)
        plt.plot(t_shown, plotaxis*i2, color=colour_I, label=label+' - shotnoise')
        if np.max(i2)>i2_max: 
            i2_max=np.max(i2)

    if xrange is not None:
        plt.xlim(xrange[0], xrange[1])
    else:
        plt.xlim(t_shown[0], t_shown[-1]+bintime)
    
    if yrange is not None:
        plt.ylim(yrange[0],yrange[1])
    else:
        if -1 in axis:
            plt.ylim(-1.5*i2_max,1.5*i2_max)
        else:
            plt.ylim(0,2*i2_max)
        
    plt.xlabel('time (s)')
    
    if bintime==1e-3:
        plt.ylabel(f'I (kHz)') 

    plt.ylabel(f'counts / {bintime * 1e3:g} ms')

    plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
    plt.title(filename+title,y=1.1,loc='left')
    plt.savefig(folder+filename+title+'.png', dpi=1200, bbox_inches = "tight")
    plt.savefig(folder+filename+title+'.svg', dpi=1200, bbox_inches = "tight")
    plt.show()


def plot_poisson(lifetimes, k, dt_E, xrange, title, folder, filename):
    plt.figure(figsize=figsize)
    lifetimes = lifetimes * dt_E * 1e3 #lifetime in dt_E * milliseconds
    
    bins = np.linspace(0, 1.1*np.max(lifetimes), 50) # in ms
    counts, _ = np.histogram(lifetimes, bins=bins);

    tau = (bins[:-1] + bins[1:]) / 2

    plt.plot(tau, counts, 'o', color='red', label='simulation')
    expected = k * np.exp(-k * tau * 1e-3)
    plt.plot(tau, expected/np.sum(expected)*np.size(lifetimes), color='black', label='Poisson')

    plt.semilogy()
    if xrange is not None:
        plt.xlim(xrange)
    else:
        plt.xlim(0, np.max(bins))
    plt.ylim(1-np.log10(np.amax(counts))*0.1, 2*np.max(counts))
    plt.xlabel(r'$\tau$ (ms)')
    plt.ylabel('counts')
    plt.legend()
    plt.title(filename+title,y=1.1,loc='left')
    plt.savefig(folder+filename+'_poisson'+title+'.png', dpi=1200, bbox_inches = "tight")
    plt.savefig(folder+filename+'_poisson'+title+'.svg', dpi=1200, bbox_inches = "tight")
    plt.show()

def plot_poisson_combined(lifetimes1,lifetimes2, k1, k2, dt_E, xrange, title, folder, filename):
    plt.figure(figsize=figsize)
    lifetimes1 = lifetimes1 * dt_E * 1e3 #lifetime in dt_E * milliseconds
    lifetimes2 = lifetimes2 * dt_E * 1e3 #lifetime in dt_E * milliseconds

    bins = np.linspace(0, 1.1*np.maximum(np.max(lifetimes1),np.max(lifetimes2)), 50) # in ms
    counts1, _ = np.histogram(lifetimes1, bins=bins);
    counts2, _ = np.histogram(lifetimes2, bins=bins);

    tau = (bins[:-1] + bins[1:]) / 2

    plt.plot(tau, counts1, 'o', markerfacecolor='none', color='k', label='simulation state 1')
    expected1 = k1 * np.exp(-k1 * tau * 1e-3)
    plt.plot(tau, expected1/np.sum(expected1)*np.size(lifetimes1), color='k', label='Poisson expected state 1')
    
    plt.plot(tau, counts2, 'o', markerfacecolor='none', color='red', label='simulation state 2')
    expected2 = k2 * np.exp(-k2 * tau * 1e-3)
    plt.plot(tau, expected2/np.sum(expected2)*np.size(lifetimes2), color='red', label='Poisson expected state 2')

    plt.semilogy()
    if xrange is not None:
        plt.xlim(xrange)
    else:
        plt.xlim(0, np.max(bins))
    plt.ylim(1-np.log10(np.amax(np.append(counts1,counts2)))*0.1, 2*np.maximum(np.max(counts1),np.max(counts2)))
    plt.xlabel(r'$\tau$ (ms)')
    plt.ylabel('counts')
    plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
    plt.title(filename+title,y=1.1,loc='left')
    plt.savefig(folder+filename+'_poisson_combined'+title+'.png', dpi=1200, bbox_inches = "tight")
    plt.savefig(folder+filename+'_poisson_combined'+title+'.svg', dpi=1200, bbox_inches = "tight")
    plt.show()
    
    
def correlate_timetags(timetagsA, timetagsB, tau):
    #dt = np.mean(np.diff(tau))
    #tau = np.append(tau, tau[-1] + dt)
    result = pcorrelate(timetagsA, timetagsB, bins=tau, normalize=True)
    return result - 1

def correlation_function(x, G0=6.0, tau_d=4e-3, aspect_ratio=5):
    G = (G0 / ((1 + x / tau_d) * (1 + aspect_ratio ** -2 * x / tau_d) ** 0.5))

    return G
    
def plot_correlation(timetags_A, timetags_B, excitation, colour, title, folder, filename):
    #tau = np.logspace(-5.5, 0, 50)
    tau = make_loglags(-6, 0, 8)
    autocorrelation = correlate_timetags(timetags_A, timetags_B, tau)
    
    tau = tau[1:] + tau[:-1] / 2  # take centers of bin
    
    model = Model(correlation_function)
    params = model.make_params()
    params['tau_d'].set(min=1, max=20)
    params['tau_d'].set(min=1e-5, max=1e-0)
    fit = model.fit(autocorrelation, params, x=tau)

    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [5, 2]})
    plt.subplots_adjust(hspace=0)
    ax[0].semilogx(tau, autocorrelation, 'o', color=colour)
    ax[1].set_xlim(1e-6, 1)
    ax[0].plot(tau, fit.best_fit, color='black')
    ax[0].set_ylabel('G-1 (-)')
    ax[0].set_ylim(np.asarray((-0.2, 1.2)) * fit.params['G0'])
    #ax[0].set_ylim(0,15)
    ax[1].set_ylabel('residual')
    ax[1].set_xlabel(r'$\tau$ (s)')
    ax[1].plot(tau, autocorrelation - fit.best_fit, 'o', color=colour)
    ax[1].plot(tau, np.zeros_like(tau), color='black')
    ax[1].set_ylim(0.05 * np.asarray((-1, 1)) * fit.params['G0'])
    ax[0].set_title(filename+title)
    plt.savefig(folder+filename+title+'.png', dpi=1200, bbox_inches = "tight")
    plt.savefig(folder+filename+title+'.svg', dpi=1200, bbox_inches = "tight")
    plt.show()
    parameters = np.empty((3, 2), dtype=float)

    for i, p in enumerate(fit.params):
        print(f'{p:15s} = {ufloat(fit.params[p].value, fit.params[p].stderr)}')
        parameters[i, 0] = fit.params[p].value
        parameters[i, 1] = fit.params[p].stderr
    
def warnings(dt, dt_E, E_static):
    if dt_E < dt:
        print('\n warning: dt_E < dt. Code will not run properly \n')
    if E_static:
        print('\n warning: E_static = True, no conformational dynamics will be simulated \n')
            
def plot_3D_motion(simname, folder, position, states, gradient=False):

    from mpl_toolkits import mplot3d #needed for plt.axes(projection="3d")
    plt.figure(figsize=figsize)
    n_steps_plotted=np.shape(position)[1]

    if gradient:
        cm = plt.get_cmap("Greys")
        col = np.stack([cm(i/(n_steps_plotted)) for i in range(n_steps_plotted)])
        col[:,:-1]=(col[:,:-1]/1.5)
        col2 = np.copy(col)
    
    else:
        col=np.zeros((n_steps_plotted,4))
        col[:,3]=1
        col2=np.copy(col)
        
    # col = blue
    col[:,2]=1
    # col2 = red
    col2[:,0]=1
    
    
    states=(states == 0)
    
    for i in range(3):
        col[:,i]=col[:,i]*states+col2[:,i]*~states
        
    ax=plt.axes(projection="3d")
    ax.scatter(position[0],position[1],position[2],facecolor=col,depthshade=0, marker=".",s=1)
    
    ax.set_xlabel('x ($\mu$m)')
    ax.set_ylabel('y ($\mu$m)')
    ax.set_zlabel('z ($\mu$m)')
    ax.dist = 11
    max_position=np.max(np.absolute(position))
    plt.locator_params(axis="both", integer=True, tight=True)
    ax.set_xlim(-max_position,max_position)
    ax.set_ylim(-max_position,max_position)
    ax.set_zlim(-max_position,max_position)
    plt.savefig(folder+simname+'3dwalk.png', dpi=1200)
    plt.savefig(folder+simname+'3dwalk.svg', dpi=1200)
    plt.show()

    

def simulate(simname='simulationname_', folder = r'C:\username\directory\\',
             excitation = 'ALEX100nsG1G1G1G1dR1R1R1R1d', sample_freq=1e7, dt=1e-5, n_steps=1e5, n_lags=2, concentration=75e-12, n_molecules=5, aspect_ratio=8,
             efficiency=[1.0,1.0], direct_exc=0, I0_514=350e3, I0_632=200e3, I_G514_background=0,I_R514_background=0,I_G632_background=0, I_R632_background=0, I_G_background=1e3, I_R_background=1e3, 
             E_static=False, R=[10e-9, 10e-9], E_states=[0.8,0.1], dt_E=1e-4, gmatrix=np.array([[0, 9], [11, 0]]), k_s=1e6, em_pos=[0,0,0], 
             immobile=False, plot_motion=False): #plot_motion currently not implemented
    
    simname_total=simname+excitation+'_000'
    
    #Note: If E_static is True then the first R and E_states value is used.
    
    import psutil
    
    
    # General warnings
    
    warnings(dt,dt_E, E_static) 
    
    
    # Saving simulation parameters
    
    parameters = list(locals().items())
    text_file = open(folder+simname+"parameters.txt", "w")
    text_file.write('\n'.join(str(par) for par in parameters))
    text_file.close()
    
    # Setting up parameters
    
    start = timeR.default_timer()
    filename = folder + simname

    n_steps=int(n_steps) 
    sample_fluorescence = np.round(dt * sample_freq, decimals=0).astype(int) # number of times we calculate the fluorescence per diffusion/intensity time step. The rounding is required due to float32/float64 inaccuracy
    sample_kinetics = np.round(dt_E/dt, decimals=0).astype(int) # number of times we calculate the diffusion/intensity per kinetics time step.
    tot_timesteps = int(n_steps * sample_fluorescence ) # if this is not changed into an int this can give the wrong number at tot_timesteps * n_lags
       
    
    # Determine transition rates from G_matrix
    
    if E_static is False:
        gmatrix = np.float32(gmatrix) #revisit
        k_0 = k_s * np.exp(-(gmatrix[0, 1] - gmatrix[0, 0])) #transition rate state 0 to state 1
        k_1 = k_s * np.exp(-(gmatrix[1, 0] - gmatrix[1, 1]))
        print(f'\n k_0: {k_0} \n k_1: {k_1} \n lifetimes state 0: {1/k_0*1e3} ms \n lifetimes state 1: {1/k_1*1e3} ms \n')
        states_first = np.random.choice([0,1], n_molecules, p=[k_1/(k_0+k_1),k_0/(k_0+k_1)])
    else:
        states_first = np.zeros(n_molecules, dtype=np.int8)
    
        
    # Determine pulse scheme
    
    pulse_time, pulse_scheme = pulses(excitation+'_000.h5') # Revisit
    pulse_length = len(pulse_scheme)
    
    
    # Initialise arrays for lifetime calculations

    lifetimes_0_total = []
    lifetimes_1_total = []
    D = np.zeros_like(R)
    for i, r in enumerate(R):
        D[i] = calc_diffusion_coefficient(r)
    box = calc_box(concentration / n_molecules, aspect_ratio)
    start_positions = np.random.uniform(low=-0.5, high=0.5, size=(n_molecules,3)) * box
    timetags_G=[]
    timetags_R=[]
    last_timetag_G = 0
    last_timetag_R = 0
    
    
    # Simulate per lag time

    for lag in range(n_lags):
        time = np.arange(start=0, stop=n_steps*dt, step=1/sample_freq) + lag * n_steps*dt
        I_G514 = np.zeros(tot_timesteps)
        I_R514 = np.zeros(tot_timesteps)
        I_G632 = np.zeros(tot_timesteps)
        I_R632 = np.zeros(tot_timesteps)
        

        if E_static is True:
            states=np.zeros((n_molecules, n_steps))
        else:
            states, states_first, lifetimes_0, lifetimes_1 = get_states(states_first, n_molecules, tot_timesteps, dt_E, k_s, gmatrix, lag, n_lags, n_steps, dt, sample_kinetics)
            lifetimes_0_total=np.append(lifetimes_0_total, lifetimes_0)
            lifetimes_1_total=np.append(lifetimes_1_total, lifetimes_1)
        E, D_array = assign_FRET_coefficient(states, E_states, D)


        

        for molecule in tqdm(range(n_molecules), desc=f'Intensity and timetags, lag {lag + 1}/{n_lags}'):
            if immobile:
                position = np.zeros((3,int(n_steps)))
                start_positions[molecule] = position[:, -1]
            else:
                position, start_positions = get_position(n_steps, molecule, dt, D_array[molecule], start_positions, box)
                
            I_G514, I_R514 = convolve_psf(position, I_G514, I_R514, efficiency, E[molecule], dir_exc=direct_exc,I0=I0_514, wavelength=514e-9, em_pos=em_pos, samples=sample_fluorescence, aspect_ratio=aspect_ratio)
            I_R632, I_G632 = convolve_psf(position, I_R632, I_G632, np.flip(efficiency), E=0, dir_exc=0, I0=I0_632, wavelength=632e-9, em_pos=em_pos, samples=sample_fluorescence, aspect_ratio=aspect_ratio)
        I_G,I_R = apply_PIE(time, I_G514, I_R514, I_G632, I_R632, 
                                               I_G514_background, I_R514_background, I_G632_background, I_R632_background, 
                                               pulse_time, pulse_scheme, pulse_length)

        timetags_G, last_timetag_G = fill_timetags(time, timetags_G, last_timetag_G, I_G, I_G_background)
        timetags_R, last_timetag_R = fill_timetags(time, timetags_R, last_timetag_R, I_R, I_R_background)
        print('\n',psutil.virtual_memory())


    if E_static != True:
        np.save(folder+simname_total+'_lifetimes_0', np.float64(lifetimes_0_total))
        np.save(folder+simname_total+'_lifetimes_1', np.float64(lifetimes_1_total))
        plot_poisson(lifetimes_0_total,k_0,dt_E, xrange=None, title='_low_energy', folder=folder, filename=simname_total)
        plot_poisson(lifetimes_1_total,k_1,dt_E, xrange=None, title='_high_energy', folder=folder, filename=simname_total)
        plot_poisson_combined(lifetimes_0_total,lifetimes_1_total, k_0, k_1, dt_E, xrange=None, title='', folder=folder, filename=simname_total)


    timetags_G514, timetags_R514, timetags_G632, timetags_R632 = select_timetags(timetags_G, timetags_R, pulse_time, pulse_scheme, pulse_length)

    saving = (timetags_G,timetags_R,timetags_G514,timetags_R514,timetags_G632,timetags_R632,I_G+I_G_background,I_R+I_R_background)
    saving = (timetags_G,timetags_R,timetags_G514,timetags_R514,timetags_G632,timetags_R632)
    savingname = ('_tt_G','_tt_R','_tt_G514','_tt_R514','_tt_G632','_tt_R632', '_I_G', '_I_R')
    for i, save in enumerate(saving):
        #(np.shape(save))
        np.save(folder+simname_total+savingname[i], np.float64(save))

    for bintime, tag, xrange, yrange in zip([0.1e-3, 1e-3, 10e-3],['100us','1ms','10ms'],[[0,0.1],[0,1],[0,1]],[[0,20],[0,150],[0,1500]]):
        #plot_timetrace(time,[timetags_G514,timetags_R514], [I_G514,I_R514], [I_G514_background,I_R514_background], axis=[1,-1], colour_tt=['green','blue'],colour_I=['grey','black'], label=['donor','acceptor'], bintime=bintime, xrange=xrange, yrange=None, title='_Time_trace_514_last_lag_'+tag+'bin', folder=folder, filename=simname_total)
        #plot_timetrace(time,[timetags_G514], [I_G514], [I_G514_background], axis=[1], colour_tt=['green'],colour_I=['grey'], label=['donor'], bintime=bintime, xrange=xrange, yrange=yrange, title='_Time_trace_G514_last_lag_'+tag+'bin', folder=folder, filename=simname_total)
        plot_timetrace(time,[timetags_R514], [I_R514], [I_R514_background], axis=[1], colour_tt=['blue'],colour_I=['black'], label=['acceptor'], bintime=bintime, xrange=xrange, yrange=yrange, title='_Time_trace_R514_last_lag_'+tag+'bin', folder=folder, filename=simname_total)

    print(timeR.default_timer() - start)
    print(f'total simulated time = {time[-1]} (s)')
    print(f'total number of photons green = {len(timetags_G)}')
    print(f'mean intensity green = {len(timetags_G) / time[-1]:.1f} (Hz)')
    print(f'total number of photons red = {len(timetags_R)}')
    print(f'mean intensity red = {len(timetags_R) / time[-1]:.1f} (Hz)')

    pulsewidth = (pulse_time * sample_freq * len(pulse_scheme))
    max_timetags = max(np.size(timetags_R), np.size(timetags_G))
    
    plot_correlation(timetags_G514, timetags_G514, excitation, colour='green', title='_G514_autocorrelation', folder=folder, filename=simname_total)


def main():

    folder=r'C:\username\directory\\'
    gmatrix=np.array([[0, 9], [9, 1]])
    R=[10e-9, 15e-9]
    
    simulate(simname='test_R20_', folder=folder,gmatrix=gmatrix, R=R)
 


if __name__ == "__main__":
    import time
    start = time.time()
    main()
    print('It took {0:0.1f} seconds'.format(time.time() - start))