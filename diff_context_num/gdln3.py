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
  out = jnp.dot(params[1][:,num_hidden*0:num_hidden*1], h1[num_hidden*0:num_hidden*1])
  return out

@jit
def predict_gated_single_context(params, inputs):
  # Propagate data forward through the network
  h1 = jnp.dot(params[0][:,:num_obj], inputs)
  # Apply general (*) module
  out = jnp.dot(params[1][:,num_hidden*0:num_hidden*1], h1[num_hidden*0:num_hidden*1])
  out = jnp.zeros(out.shape)
  # Apply module a -> context 1
  m1 = jnp.dot(params[1][:,num_hidden*1:num_hidden*2],h1[num_hidden*1:num_hidden*2,num_obj*0:num_obj*1])
  out = out.at[:,:num_obj].add(m1)
  # Apply module b -> context 2
  m2 = jnp.dot(params[1][:,num_hidden*2:num_hidden*3],h1[num_hidden*2:num_hidden*3,num_obj*1:num_obj*2])
  out = out.at[:,num_obj:num_obj*2].add(m2)
  # Apply module c -> context 3
  m3 = jnp.dot(params[1][:,num_hidden*3:num_hidden*4],h1[num_hidden*3:num_hidden*4,num_obj*2:num_obj*3])
  out = out.at[:,num_obj*2:num_obj*3].add(m3)
  return out

@jit
def predict_gated_single_context_modules(params, inputs):
  # Propagate data forward through the network
  h1 = jnp.dot(params[0][:,:num_obj], inputs)
  # Apply general (*) module
  out = jnp.dot(params[1][:,num_hidden*0:num_hidden*1], h1[num_hidden*0:num_hidden*1])
  out = jnp.zeros(out.shape)
  # Apply module a -> context 1
  m1 = jnp.dot(params[1][:,num_hidden*1:num_hidden*2],h1[num_hidden*1:num_hidden*2,num_obj*0:num_obj*1])
  out = out.at[:,:num_obj].add(m1)
  # Apply module b -> context 2
  m2 = jnp.dot(params[1][:,num_hidden*2:num_hidden*3],h1[num_hidden*2:num_hidden*3,num_obj*1:num_obj*2])
  out = out.at[:,num_obj:num_obj*2].add(m2)
  # Apply module c -> context 3
  m3 = jnp.dot(params[1][:,num_hidden*3:num_hidden*4],h1[num_hidden*3:num_hidden*4,num_obj*2:num_obj*3])
  out = out.at[:,num_obj*2:num_obj*3].add(m3)
  return [m1,m2,m3]

@jit
def predict_gated_double_context(params, inputs):
  # Propagate data forward through the network
  h1 = jnp.dot(params[0][:,:num_obj], inputs)
  # Apply general (*) module
  out = jnp.dot(params[1][:,num_hidden*0:num_hidden*1], h1[num_hidden*0:num_hidden*1])
  out = jnp.zeros(out.shape)
  # Apply module d -> context 1,2
  m4 = jnp.dot(params[1][:,num_hidden*4:num_hidden*5],h1[num_hidden*4:num_hidden*5,num_obj*0:num_obj*2])
  out = out.at[:,:num_obj*2].add(m4)
  # Apply module e -> context 2,3
  m5 = jnp.dot(params[1][:,num_hidden*5:num_hidden*6],h1[num_hidden*5:num_hidden*6,num_obj*1:num_obj*3])
  out = out.at[:,num_obj:num_obj*3].add(m5)
  # Apply module f -> context 1,3
  m6 = jnp.dot(params[1][:,num_hidden*6:num_hidden*7],h1[num_hidden*6:num_hidden*7,num_obj*0:num_obj*1])
  out = out.at[:,:num_obj].add(m6)
  m7 = jnp.dot(params[1][:,num_hidden*6:num_hidden*7],h1[num_hidden*6:num_hidden*7,num_obj*2:num_obj*3])
  out = out.at[:,num_obj*2:num_obj*3].add(m7)
  return out

@jit
def predict_gated_double_context_modules(params, inputs):
  # Propagate data forward through the network
  h1 = jnp.dot(params[0][:,:num_obj], inputs)
  # Apply general (*) module
  out = jnp.dot(params[1][:,num_hidden*0:num_hidden*1], h1[num_hidden*0:num_hidden*1])
  out = jnp.zeros(out.shape)
  # Apply module d -> context 1,2
  m4 = jnp.dot(params[1][:,num_hidden*4:num_hidden*5],h1[num_hidden*4:num_hidden*5,num_obj*0:num_obj*2])
  out = out.at[:,:num_obj*2].add(m4)
  # Apply module e -> context 2,3
  m5 = jnp.dot(params[1][:,num_hidden*5:num_hidden*6],h1[num_hidden*5:num_hidden*6,num_obj*1:num_obj*3])
  out = out.at[:,num_obj:num_obj*3].add(m5)
  # Apply module f -> context 1,3
  m6 = jnp.dot(params[1][:,num_hidden*6:num_hidden*7],h1[num_hidden*6:num_hidden*7,num_obj*0:num_obj*1])
  out = out.at[:,num_obj*0:num_obj*1].add(m6)
  m7 = jnp.dot(params[1][:,num_hidden*6:num_hidden*7],h1[num_hidden*6:num_hidden*7,num_obj*2:num_obj*3])
  out = out.at[:,num_obj*2:num_obj*3].add(m7)
  return [m4,m5,jnp.hstack([m6,m7])]

@jit
def predict_gated_context(params, inputs):
  return predict_gated_single_context(params, inputs) + predict_gated_double_context(params, inputs)
  #return predict_gated_double_context(params, inputs)

@jit
def predict_gated_context_modules(params, inputs):
  return predict_gated_single_context_modules(params, inputs) + predict_gated_double_context_modules(params, inputs)
  #return predict_gated_double_context_modules(params, inputs)

@jit
def predict_gated(params, inputs):
  return predict_gated_common(params, inputs) + predict_gated_context(params, inputs[:num_obj])

@jit
def predict_gated_hidden(params, inputs):
  # Propagate data forward through the network
  h1 = jnp.dot(params[0], inputs)
  hidden = jnp.zeros(h1.shape)
  # All context module
  hidden = hidden.at[num_hidden*0:num_hidden*1].set(h1[num_hidden*0:num_hidden*1])
  # Single context modules
  hidden = hidden.at[num_hidden*1:num_hidden*2,num_obj*0:num_obj*1].set(h1[num_hidden*1:num_hidden*2,num_obj*0:num_obj*1])
  hidden = hidden.at[num_hidden*2:num_hidden*3,num_obj*1:num_obj*2].set(h1[num_hidden*2:num_hidden*3,num_obj*1:num_obj*2])
  hidden = hidden.at[num_hidden*3:num_hidden*4,num_obj*2:num_obj*3].set(h1[num_hidden*3:num_hidden*4,num_obj*2:num_obj*3])
  # Double context modules
  hidden = hidden.at[num_hidden*4:num_hidden*5,num_obj*0:num_obj*2].set(h1[num_hidden*4:num_hidden*5,num_obj*0:num_obj*2])
  hidden = hidden.at[num_hidden*5:num_hidden*6,num_obj*1:num_obj*3].set(h1[num_hidden*5:num_hidden*6,num_obj*1:num_obj*3])
  hidden = hidden.at[num_hidden*6:num_hidden*7,num_obj*0:num_obj*1].set(h1[num_hidden*6:num_hidden*7,num_obj*0:num_obj*1])
  hidden = hidden.at[num_hidden*6:num_hidden*7,num_obj*2:num_obj*3].set(h1[num_hidden*6:num_hidden*7,num_obj*2:num_obj*3])
  return hidden
  
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
  preds_context = predict_gated_context(params, inputs[:num_obj])
  #preds = predict_gated(params, inputs)
  return (1/2)*jnp.sum(jnp.power(preds_context - (targets - common_pred),2))

@jit
def statistics_gated(params, batch):
  inputs, targets = batch
  preds = predict_gated(params, inputs)
  return (1/2)*jnp.sum(jnp.power(preds - targets,2))
  
if __name__ == "__main__":

    @jit
    def update_gated(params, batch):
        grads_common = grad(loss_gated_common)(params, batch)
        preds_common = predict_gated_common(params, batch[0])
        grads_context = grad(loss_gated_context)(params, batch, preds_common)
        return [w - step_size * dw_com - step_size * dw_con for w,dw_com,dw_con in zip(params, grads_common,grads_context)]
    
    num_obj = 8
    X,Y = gen_data3(num_obj, diff_struct = False)
    new_cmap = combine_cmaps(plt.cm.RdGy_r, plt.cm.BrBG)

    # Training hyper-parameters
    run_idx = 0
    num_hidden = 100 #60 #8.0 #20.0 (for GDLN work using random weights and compression to share latent)
    num_modules = (1,7,1)
    num_hidden_total = num_hidden*num_modules[1]
    layer_sizes_gated = [num_obj+3, int(num_hidden), (2*num_obj-1)*4]
    param_scale_base = 0.0005/float(num_hidden_total) #0.005/float(num_hidden) #0.1/float(num_hidden) #0.00001/float(num_hidden)
    # scales for li to li+1 modules for every layer. Dimensions format: layers, input block axis, output block axis. #1.1 and 0.05
    param_scale_gated = [ [[1e1*param_scale_base, 2e1*param_scale_base, 2e1*param_scale_base, 2e1*param_scale_base,\
                                              2e1*param_scale_base, 2e1*param_scale_base, 2e1*param_scale_base]] ,
                        [[1e1*param_scale_base], [2e1*param_scale_base], [2e1*param_scale_base], [2e1*param_scale_base],\
                                             [2e1*param_scale_base], [2e1*param_scale_base], [2e1*param_scale_base]] ] 
    num_epochs = 8001 #12000
    step_size = 0.001 #0.01
    seed = np.random.randint(0,100000) # can set seed here, for now it is random. The only randomness is in the network init

    losses = np.zeros( num_epochs )
    shared_svs = np.zeros( (num_epochs, num_obj+3) )
    svs_a = np.zeros( (num_epochs, num_obj+3) )
    svs_b = np.zeros( (num_epochs, num_obj+3) )
    svs_c = np.zeros( (num_epochs, num_obj+3) )
    svs_d = np.zeros( (num_epochs, num_obj+3) )
    svs_e = np.zeros( (num_epochs, num_obj+3) )
    svs_f = np.zeros( (num_epochs, num_obj+3) )

    params_gated = init_random_params_gated(param_scale_gated, layer_sizes_gated, num_modules, seed)
    for epoch in range(num_epochs):
        params_gated = update_gated(params_gated, (X,Y))
        losses[epoch] = statistics_gated(params_gated, (X,Y))
 
        if (epoch % 100) == 0:
            print('Epoch: ',epoch,'Gated Loss: ', losses[epoch])

        if epoch in [2000]:
            plot_outputs(predict_gated_double_context_modules(params_gated, X[:num_obj])[0], new_cmap, 'plots/gdln3_cont_out'+str(epoch)+'.pdf', vmin=-1, vmax=1)
            plot_outputs(predict_gated(params_gated, X), new_cmap, 'plots/gdln3_out'+str(epoch)+'.pdf', vmin=-1, vmax=1)
        
        U_net, shared_svs[epoch], VT_net = np.linalg.svd(jnp.dot(params_gated[1][:,:num_hidden], params_gated[0][:num_hidden]),\
                                                                                                full_matrices=False)
        svs_a[epoch] = np.linalg.svd(jnp.dot(params_gated[1][:,num_hidden*1:num_hidden*2],\
                                             params_gated[0][num_hidden*1:num_hidden*2]), full_matrices=False, compute_uv=False)
        svs_b[epoch] = np.linalg.svd(jnp.dot(params_gated[1][:,num_hidden*2:num_hidden*3],\
                                             params_gated[0][num_hidden*2:num_hidden*3]), full_matrices=False, compute_uv=False)
        svs_c[epoch] = np.linalg.svd(jnp.dot(params_gated[1][:,num_hidden*3:num_hidden*4],\
                                             params_gated[0][num_hidden*3:num_hidden*4]), full_matrices=False, compute_uv=False)
        svs_d[epoch] = np.linalg.svd(jnp.dot(params_gated[1][:,num_hidden*4:num_hidden*5],\
                                             params_gated[0][num_hidden*4:num_hidden*5]), full_matrices=False, compute_uv=False)
        svs_e[epoch] = np.linalg.svd(jnp.dot(params_gated[1][:,num_hidden*5:num_hidden*6],\
                                             params_gated[0][num_hidden*5:num_hidden*6]), full_matrices=False, compute_uv=False)
        svs_f[epoch] = np.linalg.svd(jnp.dot(params_gated[1][:,num_hidden*6:num_hidden*7],\
                                                   params_gated[0][num_hidden*6:num_hidden*7]), full_matrices=False, compute_uv=False)

    # Num epoch, num_data, hidden_dim
    np.savetxt('losses/gated3.txt', losses)
    np.savetxt('svs3/gated3_shared.txt', shared_svs)
    np.savetxt('svs3/gated3_a.txt', svs_a)
    np.savetxt('svs3/gated3_b.txt', svs_b)
    np.savetxt('svs3/gated3_c.txt', svs_c)
    np.savetxt('svs3/gated3_d.txt', svs_d)
    np.savetxt('svs3/gated3_e.txt', svs_e)
    np.savetxt('svs3/gated3_f.txt', svs_f)
