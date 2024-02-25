import pennylane as qml
import torch

import qiskit
import qiskit.providers.aer.noise as noise
import pdb
#from pennylane_cirq import ops as cirq_ops
# from pennylane_ionq import ops # not used right now, but for future
from pennylane import numpy as np # autograd compatible numpy
from pennylane.numpy import pi 
import matplotlib.pyplot as plt
from torch.autograd.functional import hessian, jacobian
plt.rcParams['figure.facecolor'] = 'w'
#control display precision

if True:    #for collapsing following block in Spyder or any editor
    import seaborn as sns
    
    sns.set_color_codes("deep"); sns.set_context("paper"); sns.set_style("ticks")
    STYLE_DICT = dict.fromkeys(['axes.labelsize', 'xtick.labelsize', 'ytick.labelsize', 'axes.titlesize'], 'medium')
    STYLE_DICT.update({'font.size': 18, 'figure.dpi':150, 'image.cmap': 'plasma'})
    STYLE_DICT.update(dict.fromkeys(['ytick.direction', 'xtick.direction'], 'in'))
    STYLE_DICT.update(dict.fromkeys(['xtick.major.width', 'ytick.major.width', 'axes.linewidth'], .5))
    plt.rcParams.update(STYLE_DICT)
    
def fidelity(cur_state, target_state): # Inner product 
    # target_state = torch.Tensor(target_state)
    # cur_state = torch.Tensor(target_state)\
    return torch.real(torch.dot(torch.conj(cur_state), target_state))

def norm(state): # normalize 
    # state = torch.Tensor(state)
    return state / torch.sqrt((torch.dot(torch.conj(state), state)))

def phi_parameterization(params): # restrict phase angles to -pi to pi
    return torch.clip(params, -1.5*pi, 1.5*pi) #np is okay since the params input should be a pytorch array


num_wires = 5
param_shape = qml.StronglyEntanglingLayers.shape(n_layers=3, n_wires=num_wires) 
weights = torch.rand(param_shape)
dev = qml.device("default.qubit", num_wires)


@qml.qnode(dev, interface='torch', diff_method="backprop")
def circuit_random_layers(parameters, noise_amp=0.00):
    rnd_phases = (torch.rand(parameters.shape) - 0.50) * noise_amp*2*pi
    qml.StronglyEntanglingLayers(weights=parameters+rnd_phases, wires=range(num_wires))
    return qml.state()

@qml.qnode(dev, interface='torch', diff_method="backprop")
def circuit_random_layers_no_noise(parameters):
    qml.StronglyEntanglingLayers(weights=parameters, wires=range(num_wires))
    return qml.state()



ghz_list = [1] + [0.0]*(2**num_wires - 2) + [-1] # GHZ state
w_list = [0]*(2**num_wires); 
for ii in range(num_wires):
    w_list[2**ii] = 1; #1/torch.sqrt(N_wires) # W state
    
#target_state = ghz_list    
target_state = norm(torch.tensor(ghz_list, dtype=torch.complex128)) # W state
Nshots = 2 # Number of shots each with a diff initial phase angles
steps = 50 # Number of steps per shot
eta = 0.09 # learning rate
phi_len = torch.prod(torch.tensor(param_shape))
I = torch.eye(phi_len, requires_grad=False); delta = 1.50; alpha = 5*eta
# random_seeds = range(101, 25000, 33) # construct a large array of seeds > Nshots in length
random_seeds = range(14, 25000, 35) # construct a large array of seeds > Nshots in length

phi_arr_best = torch.zeros(phi_len, requires_grad=False)
costs_final = 1
cost_history_best = []
noise_amplitude = 0.000

def cost_fn_no_noise(phi_arr): # define local cost function with only parameters as phase angles
    return 1. - \
        fidelity(circuit_random_layers_no_noise(phi_parameterization(phi_arr.reshape(param_shape))), target_state)

def cost_fn_noisy(phi_arr): # define local cost function with only parameters as phase angles
    return 1. - \
        fidelity(circuit_random_layers(phi_parameterization(phi_arr.reshape(param_shape)), noise_amp=noise_amplitude), target_state)


for ii in range (Nshots):
    print(f"\nEpoch {ii}")
    torch.manual_seed(random_seeds[ii])
    phi_arr0 = torch.randn(phi_len, requires_grad=True)
    phi_arr = phi_arr0
    K = torch.zeros((phi_len, phi_len), requires_grad=False)
    cost_history = []

    optimizer = torch.optim.SGD([phi_arr], lr=eta*10.5)
    # optimizer = torch.optim.LBFGS([phi_arr], lr=eta/3, tolerance_change=1e-13, tolerance_grad=1e-13)
    def closure():
        print(cost_fn_noisy(phi_arr))
        optimizer.zero_grad()
        loss = cost_fn_noisy(phi_arr)
        loss.backward()
        return loss

    for jj in range(steps):
        optimizer.step(closure)
        cost = cost_fn_noisy(phi_arr)
        if (jj+1) % (steps//5) == 0 or jj==steps-1:
            print(f"Step {jj+1:3d}\t Cost_L = {cost:0.7f}")
        cost_history.append(cost.clone().detach())
    if ii == 0:
        costs_final = cost_history[-1]
        cost_history_best = np.copy(cost_history)
        phi_arr_best = phi_arr.clone().detach()
    if costs_final > cost_history[-1]:
        costs_final = cost_history[-1]
        cost_history_best = np.copy(cost_history)
        phi_arr_best = phi_arr.clone().detach()
SGD_best = np.copy(cost_history_best)
print(f"Best cost out of {Nshots} attempts: {np.mean(cost_history_best[steps*9//10:])}") # For noisy final costs, take a mean over last 10% of costs
print(f"Optimized phases (in $\pi$ units): {phi_arr_best/pi}")
torch.save(cost_history_best, f"costs_IPG_delta{delta}_alpha{alpha}.pt")


