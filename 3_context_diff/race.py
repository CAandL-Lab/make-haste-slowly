import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
from jax import jit, grad, jacrev, random
import jax
import jax.numpy as jnp
import sys
from gen_data import gen_data3
from replot import plot_outputs

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=200)
matplotlib.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

def pred_dynamics(Y,X,a_init,num_svs,num_epochs):
    # Getting SVs in numpy arrays, setting up time indices array and exponential components of SV dynamics formula
    tau = 1/(X.shape[1]*step_size)

    # Init Layer SVs
    B = np.zeros((num_epochs+1,num_svs))
    Bd = np.zeros((num_epochs+1,num_svs))
    Be = np.zeros((num_epochs+1,num_svs))
    Bf = np.zeros((num_epochs+1,num_svs))

    B[0] = a_init
    Bd[0] = a_init*2e4
    Be[0] = a_init*2e4
    Bf[0] = a_init*2e4

    # Construct Pathway Gates
    gdx = jnp.zeros_like(X)
    gdx = gdx.at[:,num_obj*0:num_obj*2].set(1)
    gdx = gdx.at[num_obj:].set(0)
    gdy = jnp.zeros_like(Y)
    gdy = gdy.at[:,num_obj*0:num_obj*2].set(1)
    gex = jnp.zeros_like(X)
    gex = gex.at[:,num_obj*1:num_obj*3].set(1)
    gex = gex.at[num_obj:].set(0)
    gey = jnp.zeros_like(Y)
    gey = gey.at[:,num_obj*1:num_obj*3].set(1)
    gfx = jnp.zeros_like(X)
    gfx = gfx.at[:,num_obj*0:num_obj*1].set(1)
    gfx = gfx.at[:,num_obj*2:num_obj*3].set(1)
    gfx = gfx.at[num_obj:].set(0)
    gfy = jnp.zeros_like(Y)
    gfy = gfy.at[:,num_obj*0:num_obj*1].set(1)
    gfy = gfy.at[:,num_obj*2:num_obj*3].set(1)
    
    # Getting Singular Values of Pathway Datasets
    U, s, VT = np.linalg.svd((1/X.shape[1])*jnp.dot(Y,X.T), full_matrices=False)
    d = np.linalg.svd((1/X.shape[1])*jnp.dot(X,X.T), full_matrices=False, compute_uv=False)
    res_Y = (Y - U @ np.diag(s/d) @ VT @ X)
    Ud, sd, VdT = np.linalg.svd((1/X.shape[1])*jnp.dot(res_Y*gdy,(X*gdx).T), full_matrices=False)
    Ue, se, VeT = np.linalg.svd((1/X.shape[1])*jnp.dot(res_Y*gey,(X*gex).T), full_matrices=False)
    Uf, sf, VfT = np.linalg.svd((1/X.shape[1])*jnp.dot(res_Y*gfy,(X*gfx).T), full_matrices=False)

    # Input covariance is the same for all pathways
    delt = np.linalg.svd((1/X.shape[1])*jnp.dot(X,X.T), full_matrices=False, compute_uv=False)
    deltdd = np.linalg.svd((1/X.shape[1])*jnp.dot(X*gdx,(X*gdx).T), full_matrices=False, compute_uv=False)#-0.01
    deltde = np.linalg.svd((1/X.shape[1])*jnp.dot(X*gex,(X*gdx).T), full_matrices=False, compute_uv=False)
    deltdf = np.linalg.svd((1/X.shape[1])*jnp.dot(X*gfx,(X*gdx).T), full_matrices=False, compute_uv=False)
    deltee = np.linalg.svd((1/X.shape[1])*jnp.dot(X*gex,(X*gex).T), full_matrices=False, compute_uv=False)#-0.01
    delted = np.linalg.svd((1/X.shape[1])*jnp.dot(X*gdx,(X*gex).T), full_matrices=False, compute_uv=False)
    deltef = np.linalg.svd((1/X.shape[1])*jnp.dot(X*gfx,(X*gex).T), full_matrices=False, compute_uv=False)
    deltff = np.linalg.svd((1/X.shape[1])*jnp.dot(X*gfx,(X*gfx).T), full_matrices=False, compute_uv=False)#-0.01
    deltfd = np.linalg.svd((1/X.shape[1])*jnp.dot(X*gdx,(X*gfx).T), full_matrices=False, compute_uv=False)
    deltfe = np.linalg.svd((1/X.shape[1])*jnp.dot(X*gex,(X*gfx).T), full_matrices=False, compute_uv=False)

    overlap_de = -jnp.eye(num_svs,num_svs)*0.45 #Ud.T@Ue
    overlap_de = overlap_de.at[0,0].set(-0.49)
    #overlap_de = overlap_de.at[1,1].set(-0.45)

    overlap_df = -jnp.eye(num_svs,num_svs)*0.465 #Ud.T@Uf
    overlap_df = overlap_df.at[0,0].set(-0.49)

    overlap_ef = -jnp.eye(num_svs,num_svs)*0.465 #Ue.T@Uf
    overlap_ef = overlap_ef.at[0,0].set(-0.49)
    #overlap_ef = overlap_ef.at[0,0].set(-0.5)
    
    overlap_ed = jnp.copy(overlap_de).T #Ue.T@Ud
    overlap_fd = jnp.copy(overlap_df).T #Uf.T@Ud
    overlap_fe = jnp.copy(overlap_ef).T #Uf.T@Ue

    for t in range(0,num_epochs):
        # Errors Calc
        common_error = s - B[t]*delt
        d_error = sd - Bd[t]*(deltdd) - overlap_de@(Be[t]*deltde) - overlap_df@(Bf[t]*deltdf)
        e_error = se - Be[t]*(deltee) - overlap_ed@(Bd[t]*delted) - overlap_ef@(Bf[t]*deltef)
        f_error = sf - Bf[t]*(deltff) - overlap_fd@(Bd[t]*deltfd) - overlap_fe@(Be[t]*deltfe)

        # Updates
        B[t+1] = B[t] + (1/tau)*2*B[t]*common_error
        Bd[t+1] = Bd[t] + (1/tau)*2*Bd[t]*d_error
        Be[t+1] = Be[t] + (1/tau)*2*Be[t]*e_error
        Bf[t+1] = Bf[t] + (1/tau)*2*Bf[t]*f_error
        if t % 10 == 1:
            print(t)

    return B[1:], U, VT, Bd[1:], Ud, VdT, Be[1:], Ue, VeT, Bf[1:], Uf, VfT, gdx, gdy, gex, gey, gfx, gfy

if __name__ == "__main__":

    num_obj = 8
    X,Y = gen_data3(num_obj, diff_struct=True)

    # Training hyper-parameters
    run_idx = 0
    num_epochs = 8001 #12000
    step_size = 0.001 #0.01
    seed = np.random.randint(0,100000) # can set seed here, for now it is random. The only randomness is in the network init
 
    print("Predicting Common (*) Module")
    sv_trajectory_plots, U, VT, sv_trajectory_plots_d, Ud, VdT, sv_trajectory_plots_e, Ue, VeT, sv_trajectory_plots_f, Uf, VfT,\
            gdx, gdy, gex, gey, gfx, gfy\
            = pred_dynamics(Y,X,3e-9,11,num_epochs)
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
    d_preds_traj = jnp.dot(jnp.einsum('ij,tjk->tik', Ud, diag_sv_trajectory_plots_d), jnp.dot(VdT, X*gdx))
    e_preds_traj = jnp.dot(jnp.einsum('ij,tjk->tik', Ue, diag_sv_trajectory_plots_e), jnp.dot(VeT, X*gex))
    f_preds_traj = jnp.dot(jnp.einsum('ij,tjk->tik', Uf, diag_sv_trajectory_plots_f), jnp.dot(VfT, X*gfx))
 
    preds_traj = jnp.copy(common_preds_traj)
    preds_traj = preds_traj.at[:].add(d_preds_traj*gdy)
    preds_traj = preds_traj.at[:].add(e_preds_traj*gey)
    preds_traj = preds_traj.at[:].add(f_preds_traj*gfy)

    loss_traj = (1/2)*jnp.sum(jnp.power(preds_traj - Y[jnp.newaxis],2).reshape(preds_traj.shape[0],-1),axis=1)

    # Num epoch, num_data, hidden_dim
    np.savetxt('losses/race.txt', loss_traj)
    np.savetxt('svs/race_shared.txt', sv_trajectory_plots)
    np.savetxt('svs/race_d.txt', sv_trajectory_plots_d)
    np.savetxt('svs/race_e.txt', sv_trajectory_plots_e)
    np.savetxt('svs/race_f.txt', sv_trajectory_plots_f)
