import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
from jax import jit, grad, jacrev, random
import jax
import jax.numpy as jnp
import sys
from gen_data import gen_data3
from replot import plot_outputs, combine_cmaps

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=200)
matplotlib.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

def init_random_params_relu(scale, layer_sizes, seed):
  # Returns a list of tuples where each tuple is the weight matrix and bias vector for a layer
  np.random.seed(seed)
  init_layers = [np.random.normal(0.0, scale, (n, m)) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
  return init_layers
  
@jit
def predict_relu(params, inputs):
  # Propagate data forward through the network
  h1 = jnp.maximum(jnp.dot(params[0], inputs), 0)
  out = jnp.dot(params[1], h1)
  return out
 
@jit
def predict_relu_hidden(params, inputs):
  # Propagate data forward through the network
  h1 = jnp.maximum(jnp.dot(params[0], inputs), 0)
  return h1

@jit
def loss_relu(params, batch):
  # Loss over a batch of data
  inputs, targets = batch
  preds = predict_relu(params, inputs)
  return (1/2)*jnp.sum(jnp.power(preds - targets,2))
 
@jit
def statistics_relu(params, batch):
  inputs, targets = batch
  preds = predict_relu(params, inputs)
  return (1/2)*jnp.sum(jnp.power(preds - targets,2))
  
if __name__ == "__main__":
   
    @jit
    def update_relu(params, batch):
        grads = grad(loss_relu)(params, batch)
        return [w - step_size * dw for w,dw in zip(params, grads)]
 
    num_obj = 8
    X,Y = gen_data3(num_obj, diff_struct=True)
    new_cmap = combine_cmaps(plt.cm.RdGy_r, plt.cm.PiYG_r) #combine_cmaps(plt.cm.RdGy_r, plt.cm.PuOr)
    
    # Training hyper-parameters
    run_idx = 0
    num_hidden = 700 #60 #8.0 #20.0 (for GDLN work using random weights and compression to share latent)
    layer_sizes_relu = [(num_obj+3), int(num_hidden), (2*num_obj-1)*4]
    param_scale_relu = 0.005/float(num_hidden) #0.1/float(num_hidden) #0.00001/float(num_hidden)
    # scales for li to li+1 modules for every layer. Dimensions format: layers, input block axis, output block axis. #1.1 and 0.05

    num_epochs = 8001 #12001
    mds_sample_rate = 10
    step_size = 0.001 #0.001
    seed = np.random.randint(0,100000) # can set seed here, for now it is random. The only randomness is in the network init

    losses = np.zeros( num_epochs )
    mds = np.zeros( (int(num_epochs/mds_sample_rate)+1,int(num_hidden),X.shape[1]) )
    
    params_relu = init_random_params_relu(param_scale_relu, layer_sizes_relu, seed)
    for epoch in range(num_epochs):
        params_relu = update_relu(params_relu, (X,Y))
        losses[epoch] = statistics_relu(params_relu, (X,Y))
        if (epoch % mds_sample_rate) == 0:
            mds[int(epoch/mds_sample_rate)] = predict_relu_hidden(params_relu, X)
 
        if (epoch % 100) == 0:
            print('Epoch: ',epoch,'Relu Loss: ',losses[epoch])
       
        if epoch in [1000,4000,8000]:
            plot_outputs(predict_relu(params_relu, X), new_cmap, 'plots/relu_out'+str(epoch)+'.pdf', vmin=-1, vmax=1)

    # Num epoch, num_data, hidden_dim
    np.savetxt('losses/relu.txt', losses)
    np.savetxt('mds/relu.txt',mds.transpose(0,2,1).reshape((int(num_epochs/mds_sample_rate)+1)*X.shape[1],-1)) 
