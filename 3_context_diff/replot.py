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
    names = [['relu','gated','gated_single','race'],['relu','gated'],[m+n for m, n in pd(['gated','race'], module_names)],[m+n for m,n in pd(['gated'],unused_module_names)]]
    
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
    for i in range(len(losses)):
        plt.plot(losses[i], color=colors[i], label=labels[i], linestyle=linestyles[i])
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.axvline(500, color='black')
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
        mds_object = MDS(normalized_stress='auto', random_state=42)
        c = np.linspace(0,1,int(np.ceil(num_epochs/jump_size))+2)[:-1][1:].astype(np.float32)
        mds = mds_object.fit_transform(hidden_units[i][::jump_size].reshape((int(np.ceil(num_epochs/jump_size)))*num_data,num_hidden))
        mds = mds.reshape((int(np.ceil(num_epochs/jump_size))),num_data,2)
        for k in range(mds.shape[1]):
            plt.scatter(mds[:,k,0], mds[:,k,1], c=c, s=2, marker='o', cmap=colors[i])
            plt.text(mds[-1,k,0] + (mds[-1,k,0] - mds[-2,k,0]), mds[-1,k,1] + (mds[-1,k,1] - mds[-2,k,1]), str(k))
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        #plt.tight_layout()
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
    svals_common = True
    svals_context = True
    svals_context_mean = True

    if loss:
        losses, file_names = load_data("losses")
        labels = ['ReLU','GDLN','GDLN Single','Race Dynamics']
        #colors = ['deeppink','blue',np.array([1.0,0.1,0.0,1.]),np.array([0.36662822,0.21722414,0.56378316,1.])]
        colors = ['deeppink','blue',np.array([1.0,0.1,0.0,1.]),'green']
        linestyles = ['-','-','-','--']
        plot_losses(losses, labels, colors, linestyles, file_name='plots/losses.pdf', add_legend=True)
    
    if mds:
        sample_size = 10
        jump_size = 10
        num_samples = int(8000/sample_size)+1
        hidden_units, file_names = load_data("mds")
        labels = ['ReLU','GDLN']
        colors = [new_cmap_pink, new_cmap_blue] #['winter','autumn'] #['plasma','viridis'] #['seismic','viridis']
        hidden_units = hidden_units.reshape(hidden_units.shape[0], num_samples, int(hidden_units.shape[1]/num_samples), hidden_units.shape[2])
        plot_mds(hidden_units, jump_size, labels, colors, file_names)

    if svals_common:
        svals, file_names = load_data("svs")
        labels = ['GDLN','Race Dynamics']
        #colors = ['blue','purple']
        colors = ['blue','green']
        linestyles = ['-','--']
        plot_svals(svals[[0,4]], labels, colors, linestyles, file_name='plots/svs_common.pdf', add_legend=False)

    if svals_context:
        svals, file_names = load_data("svs")
        #print(svals[[1,2,3,5,6,7,9,10,11],-1,:7])
        print(svals[[1,2,3,5,6,7],-1,:7])
        labels = ['GDLN','Race Dynamics']
        colors = ['blue','green']
        linestyles = ['-','--']
        plot_svals(svals[[1,2,3,5,6,7]], labels, colors, linestyles, file_name='plots/svs_context.pdf', add_legend=False)

    if svals_context_mean:
        svals, file_names = load_data("svs")
        labels = ['GDLN','Race Dynamics']
        colors = ['blue','green']
        linestyles = ['-','--']
        plot_svals(np.dstack([np.mean(svals[[1,2,3]],axis=0),np.mean(svals[[5,6,7]],axis=0)]).transpose(2,0,1),\
                   labels, colors, linestyles, file_name='plots/svs_context_mean.pdf', add_legend=False) 
