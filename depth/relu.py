import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
from jax import jit, grad, jacrev, random
import jax
import jax.numpy as jnp
import sys
from sklearn.cluster import KMeans
from gen_data import gen_data3
from replot import plot_outputs, combine_cmaps

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=200)
matplotlib.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
from replot import plot_outputs

def init_random_params_relu(scale, layer_sizes, seed):
  # Returns a list of tuples where each tuple is the weight matrix and bias vector for a layer
  np.random.seed(seed)
  init_layers = [np.random.normal(0.0, scale, (n, m)) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
  return init_layers
  
@jit
def predict_relu(params, inputs):
  # Propagate data forward through the network
  x = inputs
  x = jnp.maximum(jnp.dot(params[0], x), 0)
  #x = jnp.dot(params[0], x)
  for layer in params[1:len(params)-1]:
      x = jnp.maximum(jnp.dot(layer, x), 0)
  out = jnp.dot(params[-1], x)
  return out
 
#@jit
def predict_relu_hidden(params, inputs, layer):
  # Propagate data forward through the network
  #h1 = jnp.maximum(jnp.dot(params[0], inputs), 0)
  h1 = jnp.maximum(jnp.dot(params[0], inputs), 0)
  h2 = jnp.maximum(jnp.dot(params[1], h1), 0)
  if layer == 1:
      return h1
  elif layer == 2:
      return h2
  else:
      return None
  
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
    X,Y = gen_data3(num_obj)
    new_cmap = combine_cmaps(plt.cm.RdGy_r, plt.cm.PiYG_r)

    # Training hyper-parameters
    run_idx = 0
    if len(sys.argv) > 1:
        run_idx = sys.argv[1]
    num_hidden = 60*7
    layer_sizes_relu = [(num_obj+3), int(num_hidden), int(num_hidden), (2*num_obj-1)*4]
    param_scale_relu = 0.0006 #0.0005
    
    num_epochs = 30001 #8001
    mds_sample_rate = 50
    step_size = 0.001 #0.005
    seed = int(run_idx)
    if run_idx == 0:
        seed = np.random.randint(0,100000) # can set seed here, for now it is random. The only randomness is in the network init

    losses = np.zeros( num_epochs )
    mds = np.zeros( (int(num_epochs/mds_sample_rate)+1,int(num_hidden),X.shape[1]) )
 
    params_relu = init_random_params_relu(param_scale_relu, layer_sizes_relu, seed) 
    for epoch in range(num_epochs):
        params_relu = update_relu(params_relu, (X,Y))
        losses[epoch] = statistics_relu(params_relu, (X,Y))
        
        if (epoch % mds_sample_rate) == 0:
            mds[int(epoch/mds_sample_rate)] = predict_relu_hidden(params_relu, X, 2)
 
        if (epoch % 100) == 0:
            print('Epoch: ',epoch,'Relu Loss: ',losses[epoch])

        if epoch in [15000, 22000, 30000]:
            plot_outputs(predict_relu(params_relu, X), new_cmap, 'plots/r'+str(epoch)+'.pdf')
    
    # Num epoch, num_data, hidden_dim
    if not run_idx == 0:
        np.savetxt('losses/'+str(run_idx)+'relu.txt', losses)
    else:
        np.savetxt('losses/relu.txt', losses)
