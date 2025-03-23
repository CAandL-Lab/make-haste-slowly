import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.manifold import MDS
from itertools import product as pd

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=200)
matplotlib.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

def combine_cmaps(map1, map2):
    cols1 = map1(np.linspace(0.0, 0.5, 128))
    cols2 = map2(np.linspace(0.5, 1.0, 128))
    cols = np.vstack([cols1, np.ones((4,4)), cols2])
    newmap = colors.LinearSegmentedColormap.from_list('new_colormap', cols)
    return newmap

def load_data(data_type):
    data_key = {"losses": 0, "mds": 1, "svs": 2, "unused_svs":3}
    module_names = ['_shared','_d','_e','_f']
    unused_module_names = ['_a','_b','_c']
    run_ids = [str(i) for i in np.arange(1,100)]
    names = [[n+m for m, n in pd(['relu','gated_off'],run_ids)],\
             ['relu','gated'],[m+n for m, n in pd(['race','gated','closed'], module_names)],[m+n for m,n in pd(['gated'],unused_module_names)]]
    #names = [['relu','gated','race'],['relu','gated'],[m+n for m, n in pd(['race','gated'], module_names)],[m+n for m,n in pd(['gated'],unused_module_names)]]

    data = []
    for name in names[data_key[data_type]]:
        data.append(np.loadtxt(data_type + '/' + name + '.txt'))
    return np.array(data), names[data_key[data_type]]

def plot_outputs(output, cmap, file_name, vmin=-1, vmax=1):
    plt.imshow(output, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    plt.grid()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(file_name, dpi=200, bbox_inches='tight')
    plt.close()

def plot_losses(losses, labels, colors, linestyles, file_name='losses.pdf', add_legend=True):
    relu_losses = losses[:int((losses.shape[0])/2)]
    gdln_losses = losses[int((losses.shape[0])/2):]
    relu_mean = np.mean(relu_losses,axis=0)
    gdln_mean = np.mean(gdln_losses,axis=0)
    relu_stereotype_idx = np.argmin(np.sum(np.power(relu_losses - relu_mean,2),axis=1))
    gdln_stereotype_idx = np.argmin(np.sum(np.power(gdln_losses - gdln_mean,2),axis=1)) #gdln_mean
    plt.plot(relu_losses[relu_stereotype_idx], color=colors[0], label=labels[0], linestyle=linestyles[0])
    plt.plot(gdln_losses[gdln_stereotype_idx], color=colors[1], label=labels[1], linestyle=linestyles[1])
    for i in range(len(gdln_losses)):
        plt.plot(relu_losses[i], color=colors[0], linestyle=linestyles[0],alpha=0.05)
        plt.plot(gdln_losses[i], color=colors[1], linestyle=linestyles[1],alpha=0.05)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.ylabel("Quadratic Loss")
    plt.xlabel("Epoch number")
    if add_legend:
        plt.legend()
    plt.tight_layout()
    plt.savefig(file_name, dpi=400, bbox_inches='tight')
    plt.close()

def plot_svals(svals, labels, colors, linestyles, file_name='svs.pdf', add_legend=True):
    groups_len = int(svals.shape[0]/len(colors))
    for i in range(svals.shape[0]):
        for j in range(svals[i].shape[1]):
            if j == 0:
                plt.plot(svals[i][:,j], color=colors[int(i/groups_len)], label=labels[int(i/groups_len)], linestyle=linestyles[int(i/groups_len)])
            else:
                plt.plot(svals[i][:,j], color=colors[int(i/groups_len)], linestyle=linestyles[int(i/groups_len)])
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.ylabel("Singular Value")
    plt.xlabel("Epoch number")
    if add_legend:
        plt.legend()
    plt.tight_layout()
    plt.savefig(file_name, dpi=400, bbox_inches='tight')
    plt.close()

def plot_svecs(X,Y,cmap, end_string=".png", vmin=-1, vmax=1):
    covar = (1/X.shape[1])*(Y @ X.T)
    U,s,VT = np.linalg.svd(covar,full_matrices=False)
    plt.imshow(U[:,:-1], vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.savefig('U'+end_string, dpi=200,bbox_inches='tight')
    plt.close()
    plt.imshow(VT[:-1], vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.savefig('VT'+end_string, dpi=200,bbox_inches='tight')
    plt.close()

def plot_mds(hidden_units, jump_size, labels, colors, file_names):
    for i in range(len(hidden_units)):
        num_epochs = hidden_units[i].shape[0]
        num_data = hidden_units[i].shape[1]
        num_hidden = hidden_units[i].shape[2]
        mds_object = MDS(normalized_stress='auto')
        c = np.linspace(0,1,(int(num_epochs/jump_size)+1)+2)[:-1][1:].astype(np.float32)
        mds = mds_object.fit_transform(hidden_units[i][::jump_size].reshape((int(num_epochs/jump_size)+1)*num_data,num_hidden))
        mds = mds.reshape((int(num_epochs/jump_size)+1),num_data,2)
        for k in range(mds.shape[1]):
            plt.scatter(mds[:,k,0], mds[:,k,1], c=c, s=2, marker='o', cmap=colors[i])
            plt.text(mds[-1,k,0] + (mds[-1,k,0] - mds[-2,k,0]), mds[-1,k,1] + (mds[-1,k,1] - mds[-2,k,1]), str(k))
        plt.colorbar()
        plt.savefig('plots/'+file_names[i]+'.pdf',dpi=200)
        plt.close() 

if __name__ == "__main__":

    plt.style.use('ggplot')
    matplotlib.rcParams.update({'font.size': 14})
    cols1 = plt.cm.PiYG_r(np.linspace(0.7, 1.0, 256))
    new_cmap_pink = colors.LinearSegmentedColormap.from_list('new_colormap_pink', cols1)
    cols2 = plt.cm.BrBG(np.linspace(0.7, 1.0, 256))
    new_cmap_blue = colors.LinearSegmentedColormap.from_list('new_colormap_blue', cols2) 

    loss = True
    mds = False
    svals_common = False
    svals_context = False
    svals_context_mean = False

    if loss:
        losses, file_names = load_data("losses")
        labels = ['ReLU','ReLN (GDLN)']
        colors = ['deeppink','blue']
        linestyles = ['-','-']
        plot_losses(losses, labels, colors, linestyles, file_name='plots/losses.pdf', add_legend=True)
    
    if mds:
        num_samples = int(8000/50)+1
        hidden_units, file_names = load_data("mds")
        labels = ['ReLU','GDLN']
        colors = ['plasma','viridis'] #['seismic','viridis']
        hidden_units = hidden_units.reshape(hidden_units.shape[0], num_samples, int(hidden_units.shape[1]/num_samples), hidden_units.shape[2])
        plot_mds(hidden_units, 4, labels, colors, file_names)

    if svals_common:
        svals, file_names = load_data("svs")
        labels = ['GDLN','Race Dynamics','Closed Dynamics']
        colors = ['blue','purple','green']
        linestyles = ['-','--','--']
        plot_svals(svals[[0,4,8]], labels, colors, linestyles, file_name='plots/svs_common.pdf', add_legend=False)
