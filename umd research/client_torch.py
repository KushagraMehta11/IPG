import pennylane as qml
import torch
import matplotlib.pyplot as plt
import argparse
import adam
from torch.autograd.functional import hessian, jacobian
from pennylane import numpy as np # autograd compatible numpy
from util import *


# Parses command line arguments
def parse_args():
    parser = argparse.ArgumentParser('PyTorch QML')
    parser.add_argument('--num_wires', type=int, default=3, help='number of wires in circuit')
    parser.add_argument('--optimizer', type=str, default='adam', help='GD, adam, nesterov, or IPG')
    
    # argparse does not support booleans, alternative means are possible to achieve same results
    parser.add_argument('--noise', type=int, default=0, help='noise (no noise = 0; noise = any nonzero int)')

    parser.add_argument('--Nshots', type=int, default=3, help='number of shots')
    parser.add_argument('--steps', type=int, default=32, help='number of steps')
    parser.add_argument('--eta', type=float, default=0.08, help='learning rate')

    return parser.parse_args()


# Main run function
def main(args):
    N_wires/res = args.num_wires
    param_shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=N_wires)
    
    # Choose target state, change this as needed
    target_state = norm(torch.tensor([1] + [0.0]*6 + [1], dtype=torch.complex128))
    
    Nshots = args.Nshots
    steps = args.steps
    eta = args.eta 
    phi_len = torch.prod(torch.tensor(param_shape))
    I = torch.eye(phi_len, requires_grad=False)
    delta = 1
    alpha = eta

    # construct a large array of seeds > Nshots in length
    random_seeds = range(101, 25000, 33)

    # Determine optimizer to be used
    if args.optimizer == "IPG":
        opt = None
    elif args.optimizer == "GD":
        opt = qml.GradientDescentOptimizer(eta)
    elif args.optimizer == "nesterov":
        opt = qml.NesterovMomentumOptimizer(eta)
    else:
        # module 'adam' has no attribute 'AdamOptimizer'
        opt = adam.AdamOptimizer(eta)

    phi_arr_best = torch.zeros(N_wires*3*2, requires_grad=False)
    costs_final = 1
    cost_history_best = []
    noise_amplitude = 0.001

    dev = qml.device("default.qubit", wires=N_wires)
    def cost_fn_no_noise(phi_arr): # define local cost function with only parameters as phase angles
        return 1. - \
            fidelity(qml.qnode(dev, interface='torch', diff_method="backprop"), 
                circuit_random_layers_no_noise(phi_parameterization(phi_arr.reshape(param_shape))), target_state, N_wires)

    def cost_fn_noisy(phi_arr): # define local cost function with only parameters as phase angles
        return 1. - \
            fidelity(qml.qnode(dev, interface='torch', diff_method="backprop"),
                circuit_random_layers(phi_parameterization(phi_arr.reshape(param_shape)), noise_amp=noise_amplitude), target_state, N_wires)

    # Determine gradient and hessian function
    if args.noise == 0:
        grad_fn = qml.grad(cost_fn_no_noise) # through autograd
        hess_fn = qml.jacobian(grad_fn)
    else:
        #------------# This is I think waht needs to be changed, and where I'm stuck changing
        # grad_fn = qml.grad(cost_fn_noisy) # through autograd
        # hess_fn = qml.jacobian(grad_fn)
        #------------#
        exit()

    for ii in range (Nshots):
        print(f"\nEpoch {ii}")
        torch.manual_seed(random_seeds[ii])
        phi_arr0 = torch.rand(phi_len, requires_grad=True)
        phi_arr = phi_arr0
        K = torch.zeros((phi_len, phi_len), requires_grad=False)

        cost_history = []
        for jj in range(steps):
            if args.optimizer == "IPG":
                hess = hessian(cost_fn_noisy, phi_arr)
                K = K   -   alpha * (torch.matmul(hess, K) - I)
                g = jacobian(cost_fn_noisy, phi_arr)
                phi_arr = phi_arr - delta*torch.matmul(K, g)
                cost = cost_fn_noisy(phi_arr)
            else:
                # Next line is for running built-in optimizers like GD, Adam, Nesterov
                phi_arr, cost = opt.step_and_cost(cost_fn_noisy, phi_arr, grad_fn=grad_fn)
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
    print(f"Best cost out of {Nshots} attempts: {np.mean(cost_history_best[steps*9//10:])}") # For noisy final costs, take a mean over last 10% of costs
    print(f"Optimized phases (in $\pi$ units): {phi_arr_best/pi}")


if __name__ == '__main__':
    args = parse_args()
    main(args)