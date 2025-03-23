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
 
def init_random_params_gated(scale, layer_sizes, num_modules, seed):
  # Returns a list of tuples where each tuple is the weight matrix and bias vector for a layer
  np.random.seed(seed)
  init_layers = [ np.vstack([ np.hstack([np.random.normal(0.0, scale[l][rep_p][rep_k], (n, m)) for rep_p in range(p) ])\
                                                                                               for rep_k in range(k) ])
                  for l, p, k, m, n in zip(range(len(layer_sizes)), num_modules[:-1], num_modules[1:], layer_sizes[:-1], layer_sizes[1:]) ]
  return init_layers

@jit
def predict_gated_common(params, inputs):
  # Propagate data forward through the network
  h1 = jnp.dot(params[0], inputs)
  # Apply general (*) module
  h2 = jnp.dot(params[1],h1)
  out = jnp.dot(params[2][:,num_hidden*0:num_hidden*4], h2[num_hidden*0:num_hidden*4]) #5,7 for double context or 1,3 maybe, 4,7 maybe!?
  return out

@jit
def predict_gated_context(params, inputs):
  # Propagate data forward through the network
  h1 = jnp.dot(params[0], inputs) #[:,:num_obj]
  #h1 = jnp.dot(params[0], inputs)
  h2 = jnp.dot(params[1],h1)
  # Apply general (*) module
  out = jnp.dot(params[2][:,num_hidden*0:num_hidden*1],h2[num_hidden*0:num_hidden*1])
  out = jnp.zeros(out.shape)
  # Apply module d -> context 1,2
  m4 = jnp.dot(params[2][:,num_hidden*4:num_hidden*5],h2[num_hidden*4:num_hidden*5,num_obj*0:num_obj*2])
  out = out.at[:,num_obj*0:num_obj*2].add(m4)
  # Apply module e -> context 2,3
  m5 = jnp.dot(params[2][:,num_hidden*5:num_hidden*6],h2[num_hidden*5:num_hidden*6,num_obj*1:num_obj*3])
  out = out.at[:,num_obj*1:num_obj*3].add(m5)
  # Apply module f -> context 1,3
  m6 = jnp.dot(params[2][:,num_hidden*6:num_hidden*7],h2[num_hidden*6:num_hidden*7,num_obj*0:num_obj*1])
  out = out.at[:,num_obj*0:num_obj*1].add(m6)
  m7 = jnp.dot(params[2][:,num_hidden*6:num_hidden*7],h2[num_hidden*6:num_hidden*7,num_obj*2:num_obj*3])
  out = out.at[:,num_obj*2:num_obj*3].add(m7)
  return out

@jit
def predict_gated_context_modules(params, inputs):
  # Propagate data forward through the network
  h1 = jnp.dot(params[0], inputs) #[:,:num_obj]
  #h1 = jnp.dot(params[0], inputs)
  h2 = jnp.dot(params[1],h1)
  # Apply general (*) module
  out = jnp.dot(params[2][:,num_hidden*0:num_hidden*1],h2[num_hidden*0:num_hidden*1])
  out = jnp.zeros(out.shape)
  # Apply module d -> context 1,2
  m4 = jnp.dot(params[2][:,num_hidden*4:num_hidden*5],h2[num_hidden*4:num_hidden*5,num_obj*0:num_obj*2])
  out = out.at[:,num_obj*0:num_obj*2].add(m4)
  # Apply module e -> context 2,3
  m5 = jnp.dot(params[2][:,num_hidden*5:num_hidden*6],h2[num_hidden*5:num_hidden*6,num_obj*1:num_obj*3])
  out = out.at[:,num_obj*1:num_obj*3].add(m5)
  # Apply module f -> context 1,3
  m6 = jnp.dot(params[2][:,num_hidden*6:num_hidden*7],h2[num_hidden*6:num_hidden*7,num_obj*0:num_obj*1])
  out = out.at[:,num_obj*0:num_obj*1].add(m6)
  m7 = jnp.dot(params[2][:,num_hidden*6:num_hidden*7],h2[num_hidden*6:num_hidden*7,num_obj*2:num_obj*3])
  out = out.at[:,num_obj*2:num_obj*3].add(m7)
  height = out.shape[0]
  return [m4,m5,jnp.hstack([m6,m7])]

@jit
def loss_gated_common(params, batch):
  # Loss over a batch of data
  inputs, targets = batch
  preds_common = predict_gated_common(params, inputs)
  #preds_context = predict_gated_context(params, inputs[:num_obj])
  #preds = predict_gated(params, inputs)
  return (1/2)*jnp.sum(jnp.power(preds_common - targets,2))

@jit
def loss_gated_context(params, batch, common_pred):
  # Loss over a batch of data
  inputs, targets = batch
  #preds_common = predict_gated_common(params, inputs)
  preds_context = predict_gated_context(params, inputs)
  #preds = predict_gated(params, inputs)
  return (1/2)*jnp.sum(jnp.power(preds_context - (targets - common_pred),2))

@jit
def statistics_gated(params, batch, t,start_t,switch_t1):
  inputs, targets = batch
  preds = predict_gated_common(params, inputs) + predict_gated_context(params, inputs)
  return (1/2)*jnp.sum(jnp.power(preds - targets,2))
  
if __name__ == "__main__":

    #@jit
    def update_gated(params, batch, t,start_t,switch_t1):
        if t < start_t:
            return [w for w in params]
        elif t < switch_t1:
            grads_common = grad(loss_gated_common)(params, batch)
            return [w - step_size * dw_com for w,dw_com in zip(params, grads_common)]
        else:
            grads_common = grad(loss_gated_common)(params, batch)
            preds_common = predict_gated_common(params, batch[0])
            grads_context = grad(loss_gated_context)(params, batch, preds_common)
            return [w - step_size * dw_com - step_size * dw_con for w,dw_com,dw_con in zip(params, grads_common,grads_context)]
    
    num_obj = 8
    X,Y = gen_data3(num_obj)
    new_cmap = combine_cmaps(plt.cm.RdGy_r, plt.cm.BrBG)

    # Training hyper-parameters
    run_idx = 0
    if len(sys.argv) > 1:
        run_idx = sys.argv[1]
    num_hidden = 60 #15* #60 #8.0 #20.0 (for GDLN work using random weights and compression to share latent)
    num_modules = (1,7,7,1)
    num_hidden_total = num_hidden*num_modules[1]
    layer_sizes_gated = [num_obj+3, int(num_hidden), int(num_hidden), (2*num_obj-1)*4]
    param_scale_base = 0.0005 
    param_scale_gated = [ [[*[1.15e0*param_scale_base for _ in range(num_modules[1])]] for _ in range(num_modules[0])] ,
                          [[*[1.1e0*param_scale_base for _ in range(num_modules[2])]] for _ in range(num_modules[1])] ,
                          [ [*[1.1e0*param_scale_base for _ in range(num_modules[3])]] for _ in range(0,2) ]+\
                          [ [*[1.1e0*param_scale_base for _ in range(num_modules[3])]] for _ in range(2,4) ]+\
                          [ [*[0.8e1*param_scale_base for _ in range(num_modules[3])]] for _ in range(4,num_modules[2]) ] ] #1.2e1
    num_epochs = 30001 #30001 #8001
    start_epoch = 500 #500
    switch_epoch1 = 12000 #10000 #190000 #14000
    mds_sample_rate = 50
    step_size = 0.001 #0.005
    seed = int(run_idx) # can set seed here, for now it is random. The only randomness is in the network init
    if run_idx == 0:
        seed = np.random.randint(0,100000)

    losses = np.zeros( num_epochs )
    shared_svs = np.zeros( (num_epochs, num_obj+3) )
    mds = np.zeros( (int(num_epochs/mds_sample_rate)+1,int(num_hidden_total),X.shape[1]) )

    params_gated = init_random_params_gated(param_scale_gated, layer_sizes_gated, num_modules, seed)
    for epoch in range(num_epochs):
        params_gated = update_gated(params_gated, (X,Y), epoch,start_epoch,switch_epoch1)
        losses[epoch] = statistics_gated(params_gated, (X,Y), epoch,start_epoch,switch_epoch1)
         
        if (epoch % 100) == 0:
            print('Epoch: ',epoch,'Gated Loss: ', losses[epoch])

        if epoch in [15000, 22000, 30000]:
            plot_outputs(predict_gated_common(params_gated,X) + predict_gated_context(params_gated,X),new_cmap,'plots/g'+str(epoch)+'.pdf')
             
    # Num epoch, num_data, hidden_dim
    if not run_idx == 0:
        np.savetxt('losses/'+str(run_idx)+'gated.txt', losses)
    else:
        np.savetxt('losses/gated_off.txt', losses)
