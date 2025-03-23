import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
from jax import jit, grad, jacrev, random
import jax
import jax.numpy as jnp
import sys
from gen_data import gen_data3

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=200)
matplotlib.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

def pred_svds(Y,X,a_init,num_svs):
    # Getting SVs in numpy arrays, setting up time indices array and exponential components of SV dynamics formula
    print("Common input shape")
    print(X.shape[1])
    tau = 1/(X.shape[1]*step_size)
    U, s, VT = np.linalg.svd((1/X.shape[1])*jnp.dot(Y,X.T), full_matrices=False)
    d = np.linalg.svd((1/X.shape[1])*jnp.dot(X,X.T), full_matrices=False, compute_uv=False)
    s = s[:num_svs]
    d = d[:num_svs]
    d_inv = 1/d
    taus = (np.arange(0,num_epochs,1).reshape(num_epochs,1)/tau)
    exp_s = np.exp(-2*s*taus)

    # Predict SV Trajectories using dynamics formula
    numerator = s*d_inv
    denominator = (1 - (1 - (s*d_inv/a_init))*exp_s)
    sv_trajectory_plots = numerator/denominator
    
    return sv_trajectory_plots, U[:,:num_svs], VT[:num_svs]

def pred_svds_mod(Y,X,a_init,num_svs):
    # Getting SVs in numpy arrays, setting up time indices array and exponential components of SV dynamics formula
    tau = 1/(24*step_size) #1/(X.shape[1]*step_size)
    #d_covar = (1/24)*jnp.dot(Y,X.T) #(1/X.shape[1])*jnp.dot(Y,X.T)
    U, s, VT = np.linalg.svd((1/24)*jnp.dot(Y,X.T), full_matrices=False) #np.linalg.svd((1/X.shape[1])*jnp.dot(Y,X.T), full_matrices=False)
    d = np.linalg.svd((1/24)*jnp.dot(X,X.T), full_matrices=False, compute_uv=False) #np.linalg.svd((1/X.shape[1])*jnp.dot(X,X.T), full_matrices=False)
    s = s[:num_svs]
    d = d[:num_svs]/2
    d_inv = 1/d
    taus = (np.arange(0,num_epochs,1).reshape(num_epochs,1)/tau)
    exp_s = np.exp(-2*s*taus)

    # Predict SV Trajectories using dynamics formula
    numerator = s*d_inv
    denominator = (1 - (1 - (s*d_inv/a_init))*exp_s)
    sv_trajectory_plots = numerator/denominator
    num_zero_dims = num_obj+3 - num_svs
    return np.hstack([sv_trajectory_plots, np.zeros((num_epochs, num_zero_dims))]),\
           np.hstack([U[:,:num_svs],np.zeros((U.shape[0], num_zero_dims))]),\
           np.vstack([VT[:num_svs], np.zeros((num_zero_dims,VT.shape[1]))])

if __name__ == "__main__":
 
    num_obj = 8
    X,Y = gen_data3(num_obj, diff_struct=True)

    # Training hyper-parameters
    run_idx = 0 
    num_epochs = 8001 #12001
    step_size = 0.001 #0.001
    seed = np.random.randint(0,100000) # can set seed here, for now it is random. The only randomness is in the network init

    print("Predicting Common (*) Module")
    sv_trajectory_plots, U, VT = pred_svds(Y,X,5e-9,11)
    common_end_Y = jnp.dot(jnp.dot(U, jnp.dot(jnp.diag(sv_trajectory_plots[-1]), VT)), X)
    print("Predicting module d -> context 1,2")
    sv_trajectory_plots_d, Ud, VdT  = pred_svds_mod(Y[:,num_obj*0:num_obj*2]\
                                         - common_end_Y[:,num_obj*0:num_obj*2],\
                                                      X[:num_obj,num_obj*0:num_obj*2],5e-5,num_obj) #5e-5
    d_end_Y = jnp.dot(jnp.dot(Ud, jnp.dot(jnp.diag(sv_trajectory_plots_d[-1]), VdT)), X[:num_obj])
    print("Predicting module e -> context 2,3")
    sv_trajectory_plots_e, Ue, VeT  = pred_svds_mod(Y[:,num_obj*1:num_obj*3]\
                                         - common_end_Y[:,num_obj*1:num_obj*3],\
                                                      X[:num_obj,num_obj*1:num_obj*3],5e-5,num_obj) #5e-5
    e_end_Y = jnp.dot(jnp.dot(Ue, jnp.dot(jnp.diag(sv_trajectory_plots_e[-1]), VeT)), X[:num_obj])
    print("Predicting module f -> context 1,3")
    sv_trajectory_plots_f, Uf, VfT  = pred_svds_mod(np.hstack([Y[:,num_obj*0:num_obj*1],Y[:,num_obj*2:num_obj*3]])\
                                         - np.hstack([common_end_Y[:,num_obj*0:num_obj*1],common_end_Y[:,num_obj*2:num_obj*3]]),\
                                           np.hstack([X[:num_obj,num_obj*0:num_obj*1],X[:num_obj,num_obj*2:num_obj*3]]),5e-5,num_obj) #5e-5
    print("Predicting loss")
    diag_sv_trajectory_plots = jnp.zeros((sv_trajectory_plots.shape[0], sv_trajectory_plots.shape[1], sv_trajectory_plots.shape[1]))
    diag_sv_trajectory_plots = diag_sv_trajectory_plots.at[:, np.arange(sv_trajectory_plots.shape[1]), np.arange(sv_trajectory_plots.shape[1])].set(sv_trajectory_plots)
    diag_sv_trajectory_plots_d = jnp.zeros((sv_trajectory_plots_d.shape[0], sv_trajectory_plots_d.shape[1], sv_trajectory_plots_d.shape[1]))
    diag_sv_trajectory_plots_d = diag_sv_trajectory_plots_d.at[:, np.arange(sv_trajectory_plots_d.shape[1]), np.arange(sv_trajectory_plots_d.shape[1])].set(
                                 sv_trajectory_plots_d)
    diag_sv_trajectory_plots_e = jnp.zeros((sv_trajectory_plots_e.shape[0], sv_trajectory_plots_e.shape[1], sv_trajectory_plots_e.shape[1]))
    diag_sv_trajectory_plots_e = diag_sv_trajectory_plots_e.at[:, np.arange(sv_trajectory_plots_e.shape[1]), np.arange(sv_trajectory_plots_e.shape[1])].set(
                                 sv_trajectory_plots_e)
    diag_sv_trajectory_plots_f = jnp.zeros((sv_trajectory_plots_f.shape[0], sv_trajectory_plots_f.shape[1], sv_trajectory_plots_f.shape[1]))
    diag_sv_trajectory_plots_f = diag_sv_trajectory_plots_f.at[:, np.arange(sv_trajectory_plots_f.shape[1]), np.arange(sv_trajectory_plots_f.shape[1])].set(
                                 sv_trajectory_plots_f)
    common_preds_traj = jnp.dot(jnp.einsum('ij,tjk->tik', U, diag_sv_trajectory_plots),jnp.dot(VT, X))
    d_preds_traj = jnp.dot(jnp.einsum('ij,tjk->tik', Ud, diag_sv_trajectory_plots_d), jnp.dot(VdT, X[:num_obj,num_obj*0:num_obj*2]))
    e_preds_traj = jnp.dot(jnp.einsum('ij,tjk->tik', Ue, diag_sv_trajectory_plots_e), jnp.dot(VeT, X[:num_obj,num_obj*1:num_obj*3]))
    f_preds_traj1 = jnp.dot(jnp.einsum('ij,tjk->tik', Uf, diag_sv_trajectory_plots_f), jnp.dot(VfT,X[:num_obj,num_obj*0:num_obj*1]))
    f_preds_traj2 = jnp.dot(jnp.einsum('ij,tjk->tik', Uf, diag_sv_trajectory_plots_f), jnp.dot(VfT,X[:num_obj,num_obj*2:num_obj*3]))

    preds_traj = jnp.copy(common_preds_traj)
    preds_traj = preds_traj.at[:,:,num_obj*0:num_obj*2].add(d_preds_traj)
    preds_traj = preds_traj.at[:,:,num_obj*1:num_obj*3].add(e_preds_traj)
    preds_traj = preds_traj.at[:,:,num_obj*0:num_obj*1].add(f_preds_traj1)
    preds_traj = preds_traj.at[:,:,num_obj*2:num_obj*3].add(f_preds_traj2)

    loss_traj = (1/2)*jnp.sum(jnp.power(preds_traj - Y[jnp.newaxis],2).reshape(preds_traj.shape[0],-1),axis=1)

    # Num epoch, num_data, hidden_dim
    np.savetxt('losses/closed.txt', loss_traj)
    np.savetxt('svs/closed_shared.txt', sv_trajectory_plots)
    np.savetxt('svs/closed_d.txt', sv_trajectory_plots_d)
    np.savetxt('svs/closed_e.txt', sv_trajectory_plots_e)
    np.savetxt('svs/closed_f.txt', sv_trajectory_plots_f)  
