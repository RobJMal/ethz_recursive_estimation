"""
Estimating the position of an object moving randomly on 
unit circle in discrete steps. 
"""

import numpy as np
from random import random 
import matplotlib.pyplot as plt 

# ----- HELPER FUNCTIONS -----
def calculate_process_noise(p_r):
    '''
    Calculates the process noise v(k-1). 

    Parameters:
        - r: Probability of returning 1 
    '''
    return np.random.choice([1, -1], 1, p=[p_r, 1-p_r])[0]

def calculate_sensor_noise(epsilon):
    '''
    Calculates the measurement noise of sensor based 
    on +/- epsilon

    Parameters:
        - epsilon: tolerance of range of noise (assuming +/-)
    '''
    return np.random.uniform(-1*epsilon, epsilon)

def calculate_next_state(state, noise, num_states):
    '''
    Calculates the next state of object from 
    previous state and previous noise 

    Defined as: x(k) = mod(x(k-1) + v(k-1), N)

    Parameters:
        - state: Current state of the system 
        - noise: Process noise
        - num_states: Number of states of the system 
    '''
    return (state + noise) % num_states

def calculate_distance_measurement(sensor_position, theta, noise):
    '''
    Model of the distance measurement. 

    Parameters:
        - L: position of measurement sensor
        - theta: position of object on circle
    '''
    return np.sqrt((sensor_position - np.cos(theta))**2 + np.sin(theta)**2) + noise

# ----- SETUP -----
N = 100
state_space = np.arange(0, N-1, 1)
position_theta = 2*np.pi*(state_space/N)

posterior_pdf = np.zeros(N)
prior_pdf = np.zeros(N)

print("State space: ", state_space)
print("")
print("Position theta: ", position_theta)

# Initialization

