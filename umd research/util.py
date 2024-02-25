import pennylane as qml
import torch
from pennylane.numpy import pi 

# Inner Product
def fidelity(cur_state, target_state):
    return torch.real(torch.dot(torch.conj(cur_state), target_state))


# Normalize (not norm)
def norm(state):
    return state / torch.sqrt((torch.dot(torch.conj(state), state)))


# Restricts phase angles to -pi to pi
def phi_parameterization(params): 
    return torch.clip(params, -1.5*pi, 1.5*pi)

# Circuit with Random Layers with Noise
def circuit_random_layers(parameters, noise_amp=0.00, N_wires=0):
    rnd_phases = (torch.rand(parameters.shape) - 0.50) * noise_amp*2*pi
    qml.StronglyEntanglingLayers(weights=parameters+rnd_phases, wires=range(N_wires))
    return qml.state()


# Circuit with Random Layers without Noise
def circuit_random_layers_no_noise(parameters, N_wires=0):
    qml.StronglyEntanglingLayers(weights=parameters, wires=range(N_wires))
    return qml.state()