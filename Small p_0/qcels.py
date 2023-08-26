""" Main routines for MM-QCELS 

Quantum complex exponential least squares (QCELS) can be used to
estimate the eigenvalues with reduced circuit depth. 

Last revision: 08/26/2023
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
from scipy.stats import truncnorm

def generate_QPE_distribution(spectrum,population,J):
    N = len(spectrum)
    dist = np.zeros(J)
    j_arr = 2*np.pi*np.arange(J)/J - np.pi
    for k in range(N):
        dist += population[k] * fejer_kernel.eval_Fejer_kernel(J,j_arr-spectrum[k])/J
    return dist

def generate_ts_distribution(T,NT,gamma):
    if gamma==0:
       ts=T*np.random.uniform(-1,1,NT)
    else:
       ts=truncnorm.rvs(-gamma, gamma, loc=0, scale=T, size=NT)
    return ts
    
def generate_Z_est(spectrum,population,tn,Nsample):
    N=len(tn)
    z=population.dot(np.exp(-1j*np.outer(spectrum,tn)))
    Re_true=(1+np.real(z))/2
    Im_true=(1+np.imag(z))/2
    Re_true=np.ones((Nsample, 1)) * Re_true
    Im_true=np.ones((Nsample, 1)) * Im_true
    Re_random=np.random.uniform(0,1,(Nsample,N))
    Im_random=np.random.uniform(0,1,(Nsample,N))
    Re=np.sum(Re_random<Re_true,axis=0)/Nsample
    Im=np.sum(Im_random<Im_true,axis=0)/Nsample
    Z_est = (2*Re-1)+1j*(2*Im-1)
    max_time = max(np.abs(tn))
    total_time = sum(np.abs(tn))
    return Z_est, total_time, max_time 

def generate_Z_est_multimodal(spectrum,population,T,NT,gamma):
    ts = generate_ts_distribution(T,NT,gamma)
    max_time = max(np.abs(ts))
    total_time = sum(np.abs(ts))
    Z_est, _ , _ =generate_Z_est(spectrum,population,ts,1)
    return Z_est, ts, total_time, max_time 


def generate_spectrum_population(eigenenergies, population, p):

    p = np.array(p)
    spectrum = eigenenergies * 0.25*np.pi/np.max(np.abs(eigenenergies))#normalize the spectrum
    q = population
    num_p = p.shape[0]
    q[0:num_p] = p/(1-np.sum(p))*np.sum(q[num_p:])
    return spectrum, q/np.sum(q)

def qcels_opt_fun(x, ts, Z_est):
    NT = ts.shape[0]
    N_x=int(len(x)/3)
    Z_fit = np.zeros(NT,dtype = 'complex_')
    for n in range(N_x):
       Z_fit = Z_fit + (x[3*n]+1j*x[3*n+1])*np.exp(-1j*x[3*n+2]*ts)
    return (np.linalg.norm(Z_fit-Z_est)**2/NT)

def qcels_opt_fun_coeff(x, ts, Z_est, x0):
    NT = ts.shape[0]
    N_x=int(len(x0)/3)
    Z_fit = np.zeros(NT,dtype = 'complex_')
    for n in range(N_x):
       Z_fit = Z_fit + (x[2*n]+1j*x[2*n+1])*np.exp(-1j*x0[3*n+2]*ts)
    return (np.linalg.norm(Z_fit-Z_est)**2/NT)

def qcels_opt_multimodal(ts, Z_est, x0, bounds = None, method = 'SLSQP'):
    fun = lambda x: qcels_opt_fun(x, ts, Z_est)
    N_x=int(len(x0)/3)
    bnds=np.zeros(6*N_x,dtype = 'float')
    for n in range(N_x):
       bnds[6*n]=-1
       bnds[6*n+1]=1
       bnds[6*n+2]=-1
       bnds[6*n+3]=1
       bnds[6*n+4]=-np.inf
       bnds[6*n+5]=np.inf
    bnds= [(bnds[i], bnds[i+1]) for i in range(0, len(bnds), 2)]
    if( bounds ):
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)
    else:
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)
    return res

def qcels_opt_coeff_first(ts, Z_est, x0, bounds = None, method = 'SLSQP'):
    ###need modify
    N_x=int(len(x0)/3)
    coeff=np.zeros(N_x*2)
    bnds=np.zeros(4*N_x,dtype = 'float')
    for n in range(N_x):
       bnds[4*n]=-1
       bnds[4*n+1]=1
       bnds[4*n+2]=-1
       bnds[4*n+3]=1
    bnds= [(bnds[i], bnds[i+1]) for i in range(0, len(bnds), 2)]
    for n in range(N_x):
       coeff[2*n]=x0[3*n]
       coeff[2*n+1]=x0[3*n+1]
    fun = lambda x: qcels_opt_fun_coeff(x, ts, Z_est, x0)    
    res=minimize(fun,coeff,method = 'SLSQP',bounds=bnds)
    x_out=x0
    for n in range(N_x):
       x_out[3*n]=res.x[2*n]
       x_out[3*n+1]=res.x[2*n+1]
    return x_out


def qcels_multimodal(spectrum, population, T_0, T, NT_0, NT, gamma, K, lambda_prior):        
    """Multi-level QCELS for systems with multimodal.

    Description: The code of using Multi-level QCELS to estimate the multiple dominant eigenvalues.

    Args: eigenvalues of the Hamiltonian: spectrum; 
    overlaps between the initial state and eigenvectors: population; 
    the depth for generating the data set: T_0mT; 
    number of data pairs: NT_0, NT; 
    gaussian cutoff constant: gamma; 
    initial guess of multiple dominant eigenvalues: lambda_prior
    Number of dominant eigenvalues: K
    
    Returns: an estimation of multiple dominant eigenvalues; 
    maximal evolution time T_{max}; 
    total evolution time T_{total}

    """
    total_time_all = 0.
    max_time_all = 0.
    N_level=int(np.log2(T/T_0))
    Z_est=np.zeros(NT,dtype = 'complex_')
    x0=np.zeros(3*K,dtype = 'float')
    Z_est, ts, total_time, max_time=generate_Z_est_multimodal(
        spectrum,population,T_0,NT_0,gamma) #Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
    total_time_all += total_time
    max_time_all = max(max_time_all, max_time)
    N_initial=10
    lambda_prior_collect=np.zeros((N_initial,len(lambda_prior)),dtype = 'float')
    lambda_prior_collect[0,:]=lambda_prior
    for n in range(N_initial-1):
        lambda_prior_collect[n+1,:]=np.random.uniform(spectrum[0],spectrum[-1],K)
    #Step up and solve the optimization problem
    Residue=np.inf
    for p in range(N_initial):#try different initial to make sure find global minimal
        lambda_prior_new=lambda_prior_collect[p,:]
        for n in range(K):
           x0[3*n:3*n+3]=np.array((np.random.uniform(0,1),0,lambda_prior_new[n]))
        x0 = qcels_opt_coeff_first(ts, Z_est, x0)
        res = qcels_opt_multimodal(ts, Z_est, x0)#Solve the optimization problem
        if res.fun<Residue:
            x0_fix=np.array(res.x)
            Residue=res.fun
    #Update initial guess for next iteration
    #Update the estimation interval
    x0=x0_fix
    bnds=np.zeros(6*K,dtype = 'float')
    for n in range(K):
       bnds[6*n]=-np.infty
       bnds[6*n+1]=np.infty
       bnds[6*n+2]=-np.infty
       bnds[6*n+3]=np.infty
       bnds[6*n+4]=x0[3*n+2]-np.pi/T_0
       bnds[6*n+5]=x0[3*n+2]+np.pi/T_0
    bnds= [(bnds[i], bnds[i+1]) for i in range(0, len(bnds), 2)]
    #Iteration
    for n_QCELS in range(N_level):
        T=T_0*2**(n_QCELS+1)
        Z_est, ts, total_time, max_time=generate_Z_est_multimodal(
            spectrum,population,T,NT,gamma) #Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
        total_time_all += total_time
        max_time_all = max(max_time_all, max_time)
        #Step up and solve the optimization problem
        res = qcels_opt_multimodal(ts, Z_est, x0, bounds=bnds)#Solve the optimization problem
        #Update initial guess for next iteration
        x0=np.array(res.x)
        #Update the estimation interval
        bnds=np.zeros(6*K,dtype = 'float')
        for n in range(K):
           bnds[6*n]=-np.infty
           bnds[6*n+1]=np.infty
           bnds[6*n+2]=-np.infty
           bnds[6*n+3]=np.infty
           bnds[6*n+4]=x0[3*n+2]-np.pi/T
           bnds[6*n+5]=x0[3*n+2]+np.pi/T
        bnds= [(bnds[i], bnds[i+1]) for i in range(0, len(bnds), 2)]
    #print(x0,'one iteration ends',T)
    return x0, total_time_all, max_time_all

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
    
    plt.plot(eigenenergies,population_raw,'b-o');plt.show()

    print("large overlap using multi-level QCELS")
    p0_array = np.array([0.8]) 
    T0 = 100
    N_test_QCELS = 10  #number of QCELS test
    N_QPE = 10  #number of QPE test
    T_list_QCELS = 11+T0/2*(np.arange(N_test_QCELS))
    T_list_QPE = 11+T0/2*(np.arange(N_QPE))
    err_QCELS=np.zeros((len(p0_array),len(T_list_QCELS)))
    err_QPE=np.zeros((len(p0_array),len(T_list_QPE)))
    cost_list_avg_QCELS = np.zeros((len(p0_array),len(T_list_QCELS)))
    cost_list_avg_QPE = np.zeros((len(p0_array),len(T_list_QPE)))
    rate_success_QCELS=np.zeros((len(p0_array),len(T_list_QCELS)))
    rate_success_QPE=np.zeros((len(p0_array),len(T_list_QPE)))
    max_T_QCELS=np.zeros((len(p0_array),len(T_list_QCELS)))
    Navg = 3 #number of trying
    err_thres_hold=0.01
    err_thres_hold_QPE=0.01
    #-----------------------------    
    for a1 in range(len(p0_array)):
        p0=p0_array[a1]
        n_success_QCELS= np.zeros(len(T_list_QCELS))
        n_success_QPE= np.zeros(len(T_list_QPE))
        for n_test in range(Navg):
            print("For p0=",p0,"For N_test=",n_test+1)
            spectrum, population = generate_spectrum_population(eigenenergies, 
                    population_raw, [p0])
            #------------------QCELS-----------------
            Nsample=50
            for ix in range(len(T_list_QCELS)):
                T = T_list_QCELS[ix]
                NT = 5
                lambda_prior = spectrum[0]
                ground_energy_estimate_QCELS, cost_list_avg_QCELS[a1,ix], max_T_QCELS[a1,ix] = \
                        qcels_largeoverlap(spectrum, population, T, NT,
                                Nsample, lambda_prior)
                err_this_run_QCELS = np.abs(ground_energy_estimate_QCELS - spectrum[0])
                err_QCELS[a1,ix] = err_QCELS[a1,ix]+np.abs(err_this_run_QCELS)
                if np.abs(err_this_run_QCELS)<err_thres_hold:
                    n_success_QCELS[ix]+=1
           
           # ----------------- QPE -----------------------
            N_try_QPE=int(15*np.ceil(1.0/p0))
            for ix in range(len(T_list_QPE)):
                T = int(T_list_QPE[ix])
                discrete_energies = 2*np.pi*np.arange(2*T)/(2*T) - np.pi 
                dist = generate_QPE_distribution(spectrum,population,2*T)
                samp = generate_cdf.draw_with_prob(dist,N_try_QPE)
                j_min = samp.min()
                ground_energy_estimate_QPE = discrete_energies[j_min]
                err_this_run_QPE = np.abs(ground_energy_estimate_QPE-spectrum[0])
                err_QPE[a1,ix] = err_QPE[a1,ix]+np.abs(err_this_run_QPE)
                if np.abs(err_this_run_QPE)<err_thres_hold_QPE:
                    n_success_QPE[ix]+=1
                cost_list_avg_QPE[a1,ix] = T*N_try_QPE
        rate_success_QCELS[a1,:] = n_success_QCELS[:]/Navg
        rate_success_QPE[a1,:] = n_success_QPE[:]/Navg
        err_QCELS[a1,:] = err_QCELS[a1,:]/Navg
        err_QPE[a1,:] = err_QPE[a1,:]/Navg
        cost_list_avg_QCELS[a1,:]=cost_list_avg_QCELS[a1,:]/Navg


    print('QCELS')
    print(rate_success_QCELS)
    print('QPE')
    print(rate_success_QPE)    
    plt.figure(figsize=(12,10))
    plt.plot(T_list_QCELS,err_QCELS[0,:],linestyle="-.",marker="o",label="error of QCELS p_0=0.8")
    plt.plot(T_list_QPE,err_QPE[0,:],linestyle="-.",marker="*",label="error of QPE p_0=0.8")
    plt.xlabel("$T_{max}$",fontsize=35)
    plt.ylabel("error($\epsilon$)",fontsize=35)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.figure(figsize=(12,10))
    plt.plot(cost_list_avg_QCELS[0,:],err_QCELS[0,:],linestyle="-.",marker="o",label="error of QCELS p_0=0.8")
    plt.plot(cost_list_avg_QPE[0,:],err_QPE[0,:],linestyle="-.",marker="*",label="error of QPE p_0=0.8")
    plt.xlabel("$T_{total}$",fontsize=35)
    plt.ylabel("error($\epsilon$)",fontsize=35) 
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=25)
    plt.show()


    #
    print("small overlap using multi-level QCELS with filtered data")
    p0_array=np.array([0.1]) #population
    p1_array=np.array([0.025]) #population
    #relative population=0.8
    T0 = 2000
    N_test_QCELS = 10  #number of QCELS test
    N_QPE = 10  #number of QPE test
    T_list_QCELS = 150+T0/4*(np.arange(N_test_QCELS))
    T_list_QPE = 150+T0*7.5*(np.arange(N_QPE))
    err_QCELS=np.zeros((len(p0_array),len(T_list_QCELS)))
    err_QPE=np.zeros((len(p0_array),len(T_list_QPE)))
    cost_list_avg_QCELS = np.zeros((len(p0_array),len(T_list_QCELS)))
    cost_list_avg_QPE = np.zeros((len(p0_array),len(T_list_QPE)))
    rate_success_QCELS=np.zeros((len(p0_array),len(T_list_QCELS)))
    rate_success_QPE=np.zeros((len(p0_array),len(T_list_QPE)))
    max_T_QCELS=np.zeros((len(p0_array),len(T_list_QCELS)))
    Navg = 3 #number of trying
    err_thres_hold=0.01
    err_thres_hold_QPE=0.01
    #-----------------------------    
    for a1 in range(len(p0_array)):
        p0=p0_array[a1]
        p1=p1_array[a1]
        n_success_QCELS= np.zeros(len(T_list_QCELS))
        n_success_QPE= np.zeros(len(T_list_QPE))
        for n_test in range(Navg):
            print("For p0=",p0," p1=", p1, "For N_test=",n_test+1)
            spectrum, population = generate_spectrum_population(eigenenergies, 
                    population_raw, [p0,p1])
            #------------------QCELS-----------------
            # heuristic estimate of relative gap
            rel_gap_idx = np.where(population>p0/2)[0][1] 
            rel_gap = spectrum[rel_gap_idx]-spectrum[0]
            d=int(20/rel_gap)
            print("d=", d, "rel_gap = ", rel_gap)
            Nsample_rough=int(500/p0**2*np.log(d))
            Nsample=int(30/p0**2*np.log(d))
            for ix in range(len(T_list_QCELS)):
                T = T_list_QCELS[ix]
                NT = 5
                err_tol_rough=rel_gap/4
                ground_energy_estimate_QCELS, cost_list_avg_QCELS[a1,ix], max_T_QCELS[a1,ix] = \
                        qcels_smalloverlap(spectrum, population, T, NT, d, rel_gap, \
                                        err_tol_rough, Nsample_rough, Nsample)
                err_this_run_QCELS = np.abs(ground_energy_estimate_QCELS - spectrum[0])
                err_QCELS[a1,ix] = err_QCELS[a1,ix]+np.abs(err_this_run_QCELS)
                if np.abs(err_this_run_QCELS)<err_thres_hold:
                    n_success_QCELS[ix]+=1
           
           # ----------------- QPE -----------------------
            N_try_QPE=int(15*np.ceil(1.0/p0))
            for ix in range(len(T_list_QPE)):
                T = int(T_list_QPE[ix])
                discrete_energies = 2*np.pi*np.arange(2*T)/(2*T) - np.pi
                dist = generate_QPE_distribution(spectrum,population,2*T)
                samp = generate_cdf.draw_with_prob(dist,N_try_QPE)
                j_min = samp.min()
                ground_energy_estimate_QPE = discrete_energies[j_min]
                err_this_run_QPE = np.abs(ground_energy_estimate_QPE-spectrum[0])
                err_QPE[a1,ix] = err_QPE[a1,ix]+np.abs(err_this_run_QPE)
                if np.abs(err_this_run_QPE)<err_thres_hold_QPE:
                    n_success_QPE[ix]+=1
                cost_list_avg_QPE[a1,ix] = T*N_try_QPE
        rate_success_QCELS[a1,:] = n_success_QCELS[:]/Navg
        rate_success_QPE[a1,:] = n_success_QPE[:]/Navg
        err_QCELS[a1,:] = err_QCELS[a1,:]/Navg
        err_QPE[a1,:] = err_QPE[a1,:]/Navg
        cost_list_avg_QCELS[a1,:]=cost_list_avg_QCELS[a1,:]/Navg


    print('QCELS')
    print(rate_success_QCELS)
    print('QPE')
    print(rate_success_QPE)    
    plt.figure(figsize=(12,10))
    plt.plot(T_list_QCELS,err_QCELS[0,:],linestyle="-.",marker="o",label="error of QCELS p_0=0.4,p_r=0.8")
    plt.plot(T_list_QPE,err_QPE[0,:],linestyle="-.",marker="*",label="error of QPE p_0=0.4, p_r=0.8")
    plt.xlabel("$T_{max}$",fontsize=35)
    plt.ylabel("error($\epsilon$)",fontsize=35)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.figure(figsize=(12,10))
    plt.plot(cost_list_avg_QCELS[0,:],err_QCELS[0,:],linestyle="-.",marker="o",label="error of QCELS p_0=0.4, p_r=0.8")
    plt.plot(cost_list_avg_QPE[0,:],err_QPE[0,:],linestyle="-.",marker="*",label="error of QPE p_0=0.4, p_r=0.8")
    plt.xlabel("$T_{total}$",fontsize=35)
    plt.ylabel("error($\epsilon$)",fontsize=35) 
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=25)
    plt.show()
