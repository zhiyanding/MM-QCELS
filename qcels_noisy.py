""" Main routines for QCELS 

Quantum complex exponential least squares (QCELS) can be used to
estimate the ground-state energy with reduced circuit depth. 

This adds the effect of global depolarizing noise.

Last revision: 12/10/2022
"""

import scipy.io as sio
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from scipy.special import erf
import cmath
import fejer_kernel
import fourier_filter
import generate_cdf


def generate_QPE_distribution(spectrum,population,J):
    N = len(spectrum)
    dist = np.zeros(J)
    j_arr = 2*np.pi*np.arange(J)/J - np.pi
    for k in range(N):
        dist += population[k] * fejer_kernel.eval_Fejer_kernel(J,j_arr-spectrum[k])/J
    return dist

def get_estimated_ground_energy_rough(d,delta,spectrum,population,Nsample,Nbatch):
    
    F_coeffs = fourier_filter.F_fourier_coeffs(d,delta)

    compute_prob_X = lambda T: generate_cdf.compute_prob_X_(T,spectrum,population)
    compute_prob_Y = lambda T: generate_cdf.compute_prob_Y_(T,spectrum,population)


    outcome_X_arr, outcome_Y_arr, J_arr = generate_cdf.sample_XY(compute_prob_X, 
                                compute_prob_Y, F_coeffs, Nsample, Nbatch) #Generate sample to calculate CDF

    total_evolution_time = np.sum(np.abs(J_arr))
    average_evolution_time = total_evolution_time/(Nsample*Nbatch)
    maxi_evolution_time=max(np.abs(J_arr[0,:]))

    Nx = 10
    Lx = np.pi/3
    ground_energy_estimate = 0.0
    count = 0
    #---"binary" search
    while Lx > delta:
        x = (2*np.arange(Nx)/Nx-1)*Lx +  ground_energy_estimate
        y_avg = generate_cdf.compute_cdf_from_XY(x, outcome_X_arr, outcome_Y_arr, J_arr, F_coeffs)#Calculate the value of CDF
        indicator_list = y_avg > (population[0]/2.05)
        ix = np.nonzero(indicator_list)[0][0]
        ground_energy_estimate = x[ix]
        Lx = Lx/2
        count += 1
    
    return ground_energy_estimate, count*total_evolution_time, maxi_evolution_time

def generate_filtered_Z_est(spectrum,population,t,x,d,delta,Nsample,Nbatch):
    
    F_coeffs = fourier_filter.F_fourier_coeffs(d,delta)
    compute_prob_X = lambda t_: generate_cdf.compute_prob_X_(t_,spectrum,population)
    compute_prob_Y = lambda t_: generate_cdf.compute_prob_Y_(t_,spectrum,population)
    #Calculate <\psi|F(H)\exp(-itH)|\psi>
    outcome_X_arr, outcome_Y_arr, J_arr = generate_cdf.sample_XY_QCELS(compute_prob_X, 
                                compute_prob_Y, F_coeffs, Nsample, Nbatch,t) #Generate samples using Hadmard test
    y_avg = generate_cdf.compute_cdf_from_XY_QCELS(x, outcome_X_arr, outcome_Y_arr, J_arr, F_coeffs) 
    total_time = np.sum(np.abs(J_arr))+t*Nsample*Nbatch
    max_time= max(np.abs(J_arr[0,:]))+t
    return y_avg, total_time, max_time


def generate_Z_est(spectrum,population,t,Nsample):
    Re=0
    Im=0
    z=np.dot(population,np.exp(-1j*spectrum*t))
    Re_true=(1+z.real)/2
    Im_true=(1+z.imag)/2
    #Simulate Hadmard test
    for nt in range(Nsample):
        if np.random.uniform(0,1)<Re_true:
           Re+=1
    for nt2 in range(Nsample):
        if np.random.uniform(0,1)<Im_true:
           Im+=1
    Z_est = complex(2*Re/Nsample-1,2*Im/Nsample-1)
    max_time = t
    total_time = t * Nsample
    return Z_est, total_time, max_time 
5

def generate_Z_est_noisy(spectrum,population,t,Nsample,alpha):
    """Generate the noisy data in the presence of a global depolarizing
    error channel.

    Args: 
        spectrum: eigenvalue
        population: (squared) overlap between the initial state and
            eigenstates
        t: propagation time
        Nsample: number of repetitions
        alpha: parameter for global depolarizing error channel. 
            Zexp(t) = e^{-alpha*|t|} * Z(t)
    """
    Re=0
    Im=0
    z=np.dot(population,np.exp(-1j*spectrum*t)) * \
            np.exp(-alpha*np.abs(t))
    Re_true=(1+z.real)/2
    Im_true=(1+z.imag)/2
    #Simulate Hadmard test
    for nt in range(Nsample):
        if np.random.uniform(0,1)<Re_true:
           Re+=1
    for nt2 in range(Nsample):
        if np.random.uniform(0,1)<Im_true:
           Im+=1
    Z_est = complex(2*Re/Nsample-1,2*Im/Nsample-1)
    max_time = t
    total_time = t * Nsample
    return Z_est, total_time, max_time 
       


def generate_spectrum_population(eigenenergies, population, p):

    p = np.array(p)
    spectrum = eigenenergies * 0.25*np.pi/np.max(np.abs(eigenenergies))#normalize the spectrum
    q = population
    num_p = p.shape[0]
    q[0:num_p] = p/(1-np.sum(p))*np.sum(q[num_p:])
    return spectrum, q/np.sum(q)

def qcels_opt_fun(x, ts, Z_est):
    NT = ts.shape[0]
    Z_fit=np.zeros(NT,dtype = 'complex_')
    Z_fit=(x[0]+1j*x[1])*np.exp(-1j*x[2]*ts)
    return (np.linalg.norm(Z_fit-Z_est)**2/NT)

def qcels_opt(ts, Z_est, x0, bounds = None, method = 'SLSQP'):

    fun = lambda x: qcels_opt_fun(x, ts, Z_est)
    if( bounds ):
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)
    else:
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)

    return res

def qcels_opt_noisy_fun(x, ts, Z_est, fitmode='amp'):
    NT = ts.shape[0]
    Z_fit=np.zeros(NT,dtype = 'complex_')
    Z_fit=(x[0]+1j*x[1])*np.exp(-1j*x[2]*ts)
    if( fitmode == 'amp' ):
        # amplification
        Z_amp = np.exp(x[3]*np.abs(ts))
        return (np.linalg.norm(Z_fit-Z_amp*Z_est)**2/NT)
    elif( fitmode == 'dmp' ):
        # damping
        Z_dmp = np.exp(-x[3]*np.abs(ts))
        return (np.linalg.norm(Z_dmp*Z_fit-Z_est)**2/NT)
    else:
        print('invalid fitmode')

def qcels_opt_noisy(ts, Z_est, x0, bounds = None, method = 'SLSQP', 
        fitmode = 'amp'):

    fun = lambda x: qcels_opt_noisy_fun(x, ts, Z_est, fitmode)
    if( bounds ):
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)
    else:
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)

    return res



def qcels_largeoverlap(spectrum, population, T, NT, Nsample, lambda_prior):
    """Multi-level QCELS for a system with a large initial overlap.

    Description: The code of using Multi-level QCELS to estimate the ground state energy for a systems with a large initial overlap

    Args: eigenvalues of the Hamiltonian: spectrum; 
    overlaps between the initial state and eigenvectors: population; 
    the depth for generating the data set: T; 
    number of data pairs: NT; 
    number of samples: Nsample; 
    initial guess of \lambda_0: lambda_prior

    Returns: an estimation of \lambda_0; 
    maximal evolution time T_{max}; 
    total evolution time T_{total}

    """
    total_time_all = 0.
    max_time_all = 0.

    N_level=int(np.log2(T/NT))
    Z_est=np.zeros(NT,dtype = 'complex_')
    tau=T/NT/(2**N_level)
    ts=tau*np.arange(NT)
    for i in range(NT):
        Z_est[i], total_time, max_time=generate_Z_est(
                spectrum,population,ts[i],Nsample) #Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
        total_time_all += total_time
        max_time_all = max(max_time_all, max_time)
    #Step up and solve the optimization problem
    x0=np.array((0.5,0,lambda_prior))
    res = qcels_opt(ts, Z_est, x0)#Solve the optimization problem
    #Update initial guess for next iteration
    ground_coefficient_QCELS=res.x[0]
    ground_coefficient_QCELS2=res.x[1]
    ground_energy_estimate_QCELS=res.x[2]
    #Update the estimation interval
    lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau) 
    lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau) 
    #Iteration
    for n_QCELS in range(N_level):
        Z_est=np.zeros(NT,dtype = 'complex_')
        tau=T/NT/(2**(N_level-n_QCELS-1)) #generate a sequence of \tau_j
        ts=tau*np.arange(NT)
        for i in range(NT):
            Z_est[i], total_time, max_time=generate_Z_est(
                    spectrum,population,ts[i],Nsample) #Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
            total_time_all += total_time
            max_time_all = max(max_time_all, max_time)
        #Step up and solve the optimization problem
        x0=np.array((ground_coefficient_QCELS,ground_coefficient_QCELS2,ground_energy_estimate_QCELS))
        bnds=((-np.inf,np.inf),(-np.inf,np.inf),(lambda_min,lambda_max)) 
        res = qcels_opt(ts, Z_est, x0, bounds=bnds)#Solve the optimization problem
        #Update initial guess for next iteration
        ground_coefficient_QCELS=res.x[0]
        ground_coefficient_QCELS2=res.x[1]
        ground_energy_estimate_QCELS=res.x[2]
        #Update the estimation interval
        lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau) 
        lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau) 

    return res, total_time_all, max_time_all


def qcels_smalloverlap(spectrum, population, T, NT, d, rel_gap, err_tol_rough, Nsample_rough, Nsample):
    """Multi-level QCELS with a filtered data set for a system with a small initial overlap.

    Description: The codes of using Multi-level QCELS and eigenvalue filter to estimate the ground state energy for
    a system with a small initial overlap

    Args: eigenvalues of the Hamiltonian: spectrum; 
    overlaps between the initial state and eigenvectors: population; 
    the depth for generating the data set: T; 
    number of data pairs: NT; 
    number of samples for constructing the eigenvalue filter: Nsample_rough; 
    number of samples for generating the data set: Nsample; 
    initial guess of \lambda_0: lambda_prior
    
    Returns: an estimation of \lambda_0; 
    maximal evolution time T_{max}; 
    total evolution time T_{total}

    """
    total_time_all = 0.
    max_time_all = 0.

    lambda_prior, total_time_prior, max_time_prior = get_estimated_ground_energy_rough(
            d,err_tol_rough,spectrum,population,Nsample_rough,Nbatch=1) #Get the rough estimation of the ground state energy
    x = lambda_prior + rel_gap/2 #center of the eigenvalue filter
    total_time_all += total_time_prior
    max_time_all = max(max_time_all, max_time_prior)
    
    N_level=int(np.log2(T/NT))
    Z_est=np.zeros(NT,dtype = 'complex_')
    tau=T/NT/(2**N_level)
    ts=tau*np.arange(NT)
    for i in range(NT):
        Z_est[i], total_time, max_time=generate_filtered_Z_est(
                spectrum,population,ts[i],x,d,err_tol_rough,Nsample_rough,Nbatch=1)#Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
        total_time_all += total_time
        max_time_all = max(max_time_all, max_time)
    #Step up and solve the optimization problem
    x0=np.array((0.5,0,lambda_prior))
    res = qcels_opt(ts, Z_est, x0)#Solve the optimization problem
    #Update initial guess for next iteration
    ground_coefficient_QCELS=res.x[0]
    ground_coefficient_QCELS2=res.x[1]
    ground_energy_estimate_QCELS=res.x[2]
    #Update the estimation interval
    lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau)
    lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau)
    #Iteration
    for n_QCELS in range(N_level):
        Z_est=np.zeros(NT,dtype = 'complex_')
        tau=T/NT/(2**(N_level-n_QCELS-1))
        ts=tau*np.arange(NT)
        for i in range(NT):
            Z_est[i], total_time, max_time=generate_filtered_Z_est(
                    spectrum,population,ts[i],x,d,err_tol_rough,Nsample,Nbatch=1)#Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
            total_time_all += total_time
            max_time_all = max(max_time_all, max_time)
        #Step up and solve the optimization problem
        x0=np.array((ground_coefficient_QCELS,ground_coefficient_QCELS2,ground_energy_estimate_QCELS))
        bnds=((-np.inf,np.inf),(-np.inf,np.inf),(lambda_min,lambda_max))
        res = qcels_opt(ts, Z_est, x0, bounds=bnds)#Solve the optimization problem
        #Update initial guess for next iteration
        ground_coefficient_QCELS=res.x[0]
        ground_coefficient_QCELS2=res.x[1]
        ground_energy_estimate_QCELS=res.x[2]
        #Update the estimation interval
        lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau)
        lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau)

    return ground_energy_estimate_QCELS, total_time_all, max_time_all


def qcels_largeoverlap_singlelevel_noisy(spectrum, population, T, NT, Nsample,
        lambda_prior, alpha, fitmode):
    """Single-level QCELS for a system with a large initial overlap.
    """
    total_time_all = 0.
    max_time_all = 0.

    N_level=int(np.log2(T/NT))
    Z_est=np.zeros(NT,dtype = 'complex_')
    tau=T/NT/(2**N_level)
    ts=tau*np.arange(NT)
    for i in range(NT):
        Z_est[i], total_time, max_time=generate_Z_est_noisy(
                spectrum,population,ts[i],Nsample,alpha) #Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
        total_time_all += total_time
        max_time_all = max(max_time_all, max_time)
    #Step up and solve the optimization problem
    x0=np.array((0.5,0,lambda_prior,alpha))
    lambda_min = lambda_prior - 0.1
    lambda_max = lambda_prior + 0.1

    bnds=((-np.inf,np.inf),(-np.inf,np.inf),(lambda_min,lambda_max),
            (alpha/5,alpha*5)) 
    res = qcels_opt_noisy(ts, Z_est, x0, bounds=bnds, fitmode='dmp')#Solve the optimization problem
    #Update initial guess for next iteration
    ground_coefficient_QCELS=res.x[0]
    ground_coefficient_QCELS2=res.x[1]
    ground_energy_estimate_QCELS=res.x[2]
    ground_beta = res.x[3]

    return res, total_time_all, max_time_all


if __name__ == "__main__":
    import scipy.io as sio
    import numpy as np
    from copy import deepcopy
    from scipy.optimize import minimize
    from matplotlib import pyplot as plt
    from scipy.special import erf
    from mpl_toolkits.mplot3d import Axes3D
    import cmath
    import matplotlib
    import hubbard_1d
    import quspin
    import fejer_kernel
    import fourier_filter
    import generate_cdf
    matplotlib.rcParams['font.size'] = 15
    matplotlib.rcParams['lines.markersize'] = 10

    num_sites = 4
    J = 1.0
    U = 10.0
    U0 = 0.0
    mu = 0.0
    N_up = num_sites // 2
    N_down = num_sites - N_up

    num_eigenstates_max = 100
    
    ham0 = hubbard_1d.generate_ham(num_sites, J, U0, mu, N_up, N_down)
    ground_state_0 = ham0.eigsh(k=1,which="SA")[1][:,0]
    
    ham = hubbard_1d.generate_ham(num_sites, J, U, mu, N_up, N_down)
    if( num_eigenstates_max > ham.shape[0] // 2):
        eigenenergies, eigenstates = ham.eigh()
    else:
        eigenenergies, eigenstates = ham.eigsh(k=num_eigenstates_max,which="SA")
    ground_state = eigenstates[:,0]
    
    population_raw = np.abs(np.dot(eigenstates.conj().T, ground_state_0))**2
    
    # plt.plot(eigenenergies,population_raw,'b-o');plt.show()

    spectrum, population = generate_spectrum_population(eigenenergies, 
            population_raw, [0.7])

    T = 100
    NT = 100
    noise_alpha = 0.01
    
    ground_energy = spectrum[0]
    print(ground_energy)
    Nsample_list = np.arange(1,6)*400
    error_amp = np.zeros_like(Nsample_list, dtype=float)
    error_dmp = np.zeros_like(Nsample_list, dtype=float)
    error_alpha_amp = np.zeros_like(Nsample_list, dtype=float)
    error_alpha_dmp = np.zeros_like(Nsample_list, dtype=float)

    lambda_prior = spectrum[0]
    for (i,Nsample) in enumerate(Nsample_list):
        print("Nsample = ", Nsample)
        res, cost, max_T = \
                qcels_largeoverlap_singlelevel_noisy(
                        spectrum, population, T, NT, 
                        Nsample, lambda_prior, noise_alpha,
                        fitmode='amp')
        theta = res.x[2]
        beta = np.real(res.x[3])
        error_amp[i] = theta-ground_energy
        error_alpha_amp[i] = beta - noise_alpha

        res, cost, max_T = \
                qcels_largeoverlap_singlelevel_noisy(
                        spectrum, population, T, NT, 
                        Nsample, lambda_prior, noise_alpha,
                        fitmode='dmp')
        theta = res.x[2]
        beta = np.real(res.x[3])
        error_dmp[i] = theta-ground_energy
        error_alpha_dmp[i] = beta - noise_alpha

    print("Nsample_list     = ", Nsample_list)
    print("error_amp        = ", error_amp)
    print("error_dmp        = ", error_dmp)
    print("error_alpha_amp  = ", error_alpha_amp)
    print("error_alpha_dmp  = ", error_alpha_dmp)
