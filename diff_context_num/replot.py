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
    data_key = {"losses": 0, "svs3": 1, "svs4": 2, "svs5": 3}
    arch_nums = ['3','4','5']
    module_names = ['_shared','_d','_e','_f']
    names = [[m+n for m, n in pd(['relu','gated','closed'],arch_nums)],\
             [m+n+p for m, n, p in pd(['gated','closed'], ['3'], module_names)],\
             [m+n+p for m, n, p in pd(['gated','closed'], ['4'], module_names)],\
             [m+n+p for m, n, p in pd(['gated','closed'], ['5'], module_names)]]

    print("Loading Data: ")
    print(names[data_key[data_type]])
    data = []
    for name in names[data_key[data_type]]:
        file_namer = data_type + '/' + name + '.txt'
        next_set = np.loadtxt(file_namer)
        print("Loading File: ", file_namer, ", with shape: ", next_set.shape)
        data.append(next_set)
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

if __name__ == "__main__":

    plt.style.use('ggplot')
    matplotlib.rcParams.update({'font.size': 14})

    loss_all = False
    loss3 = True
    loss4 = True
    loss5 = True
    svals_common3 = True
    svals_common4 = True
    svals_common5 = True
    svals_context3 = True
    svals_context4 = True
    svals_context5 = True
    svals_context_means3 = True
    svals_context_means4 = True
    svals_context_means5 = True

    if loss_all:
        losses, file_names = load_data("losses")
        labels = ['ReLU 3 Context','ReLU 4 Context','ReLU 5 Context',\
                  'GDLN 3 Context','GDLN 4 Context','GDLN 5 Context',\
                  'Closed Dynamics 3 Context','Closed Dynamics 4 Context','Closed Dynamics 5 Context']
        colors = ['deeppink','deeppink','deeppink',\
                  'blue','blue','blue',\
                  'green','green','green']
        linestyles = ['-','-','-','-','-','-','--','--','--']
        plot_losses(losses, labels, colors, linestyles, file_name='plots/losses.pdf', add_legend=True)

    if loss3:
        losses, file_names = load_data("losses")
        labels = ['ReLU 3 Context','GDLN 3 Context','Closed Dynamics 3 Context']
        colors = ['deeppink','blue','green']
        linestyles = ['-','-','--']
        plot_losses(losses[[0,3,6]], labels, colors, linestyles, file_name='plots/losses3.pdf', add_legend=True)

    if loss4:
        losses, file_names = load_data("losses")
        labels = ['ReLU 4 Context','GDLN 4 Context','Closed Dynamics 4 Context']
        colors = ['deeppink','blue','green']
        linestyles = ['-','-','--']
        plot_losses(losses[[1,4,7]], labels, colors, linestyles, file_name='plots/losses4.pdf', add_legend=True)

    if loss5:
        losses, file_names = load_data("losses")
        labels = ['ReLU 5 Context','GDLN 5 Context','Closed Dynamics 5 Context']
        colors = ['deeppink','blue','green']
        linestyles = ['-','-','--']
        plot_losses(losses[[2,5,8]], labels, colors, linestyles, file_name='plots/losses5.pdf', add_legend=True)

    if svals_common3:
        svals, file_names = load_data("svs3")
        labels = ['GDLN 3 Common','Closed Dynamics 3 Common',]
        colors = ['blue','green']
        linestyles = ['-','--']
        plot_svals(svals[[0,4]], labels, colors, linestyles, file_name='plots/svs_common3.pdf', add_legend=False)

    if svals_common4:
        svals, file_names = load_data("svs4")
        labels = ['GDLN 4 Common','Closed Dynamics 4 Common',]
        colors = ['blue','green']
        linestyles = ['-','--']
        plot_svals(svals[[0,4]], labels, colors, linestyles, file_name='plots/svs_common4.pdf', add_legend=False) 

    if svals_common5:
        svals, file_names = load_data("svs5")
        labels = ['GDLN 5 Common','Closed Dynamics 5 Common',]
        colors = ['blue','green']
        linestyles = ['-','--']
        plot_svals(svals[[0,4]], labels, colors, linestyles, file_name='plots/svs_common5.pdf', add_legend=False)

    if svals_context3:
        svals, file_names = load_data("svs3")
        labels = ['GDLN 3 D','GDLN 3 E','GDLN 3 F','Closed Dynamics 3 D','Closed Dynamics 3 E','Closed Dynamics 3 F']
        colors = ['blue','blue','blue','green','green','green']
        linestyles = ['-','-','-','--','--','--']
        plot_svals(svals[[1,2,3,5,6,7]], labels, colors, linestyles, file_name='plots/svs_context3.pdf', add_legend=False)

    if svals_context4:
        svals, file_names = load_data("svs4")
        labels = ['GDLN 4 D','GDLN 4 E','GDLN 4 F','Closed Dynamics 4 D','Closed Dynamics 4 E','Closed Dynamics 4 F']
        colors = ['blue','blue','blue','green','green','green']
        linestyles = ['-','-','-','--','--','--']
        plot_svals(svals[[1,2,3,5,6,7]], labels, colors, linestyles, file_name='plots/svs_context4.pdf', add_legend=False)
 
    if svals_context5:
        svals, file_names = load_data("svs5")
        labels = ['GDLN 5 D','GDLN 5 E','GDLN 5 F','Closed Dynamics 5 D','Closed Dynamics 5 E','Closed Dynamics 5 F']
        colors = ['blue','blue','blue','green','green','green']
        linestyles = ['-','-','-','--','--','--']
        plot_svals(svals[[1,2,3,5,6,7]], labels, colors, linestyles, file_name='plots/svs_context5.pdf', add_legend=False)

    if svals_context_means3:
        svals, file_names = load_data("svs3")
        labels = ['GDLN 3 Context','Closed Dynamics 3 Context']
        colors = ['blue','green']
        linestyles = ['-','--','--']
        plot_svals(np.dstack([np.mean(svals[[1,2,3]],axis=0),np.mean(svals[[5,6,7]],axis=0)]).transpose(2,0,1),\
                   labels, colors, linestyles, file_name='plots/svs_context_mean3.pdf', add_legend=False)

    if svals_context_means4:
        svals, file_names = load_data("svs4")
        labels = ['GDLN 4 Context','Closed Dynamics 4 Context']
        colors = ['blue','green']
        linestyles = ['-','--','--']
        plot_svals(np.dstack([np.mean(svals[[1,2,3]],axis=0),np.mean(svals[[5,6,7]],axis=0)]).transpose(2,0,1),\
                   labels, colors, linestyles, file_name='plots/svs_context_mean4.pdf', add_legend=False)

    if svals_context_means5:
        svals, file_names = load_data("svs5")
        labels = ['GDLN 5 Context','Closed Dynamics 5 Context']
        colors = ['blue','green']
        linestyles = ['-','--','--']
        plot_svals(np.dstack([np.mean(svals[[1,2,3]],axis=0),np.mean(svals[[5,6,7]],axis=0)]).transpose(2,0,1),\
                   labels, colors, linestyles, file_name='plots/svs_context_mean5.pdf', add_legend=False)
