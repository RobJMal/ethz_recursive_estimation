"""
Estimating the position of an object moving randomly on 
unit circle in discrete steps. 
"""

import numpy as np
import matplotlib.pyplot as plt 

N_NUM_STATES = 100
L_SENSOR_DISTANCE = 2.0
R_PROCESS_NOISE_PDF = 0.50
E_SENSOR_NOISE_TOLERANCE = 0.50
K_RECURSIVE_STEPS = 5

# ----- HELPER FUNCTIONS -----
def calculate_process_noise(p_r):
    '''
    Calculates the process noise v(k-1). 

    Parameters:
        - r: Probability of returning 1 
    '''
    return np.random.choice([1, -1], 1, p=[p_r, 1-p_r])[0]

def calculate_next_state(state, num_states):
    '''
    Calculates the next state of object from 
    previous state and previous noise 

    Defined as: x(k) = mod(x(k-1) + v(k-1), N)

    Parameters:
        - state: Current state of the system 
        - noise: Process noise
        - num_states: Number of states of the system 
    '''
    process_noise = calculate_process_noise(R_PROCESS_NOISE_PDF)
    return (state + process_noise) % num_states

def calculate_sensor_noise(epsilon):
    '''
    Calculates the measurement noise of sensor based 
    on +/- epsilon

    Parameters:
        - epsilon: tolerance of range of noise (assuming +/-)
    '''
    return np.random.uniform(-1*epsilon, epsilon)

def calculate_distance_measurement(sensor_position, theta):
    '''
    Model of the distance measurement. 

    Parameters:
        - L: position of measurement sensor
        - theta: position of object on circle
    '''
    sensor_noise = calculate_sensor_noise(E_SENSOR_NOISE_TOLERANCE)
    return np.sqrt((sensor_position - np.cos(theta))**2 + np.sin(theta)**2) + sensor_noise

def calculate_process_model_PDF(state, process_noise_pdf, num_states):
    '''
    Calculates PDF of process model 
    '''
    x_k = calculate_next_state(state, num_states)

    if x_k == ((state + 1) % num_states):
        return process_noise_pdf
    elif x_k == ((state - 1) % num_states):
        return 1 - process_noise_pdf

    return 0

def calculate_sensor_model_PDF(sensor_noise_tolerance, sensor_position, theta):
    '''
    Calculates PDF of sensor model
    '''
    z_k = calculate_distance_measurement(sensor_position, theta)

    if np.abs(z_k - np.sqrt((sensor_position - np.cos(theta))**2 - np.sin(theta)**2)) <= sensor_noise_tolerance:
        return 1/(2*sensor_noise_tolerance)

    return 0

# ----- SETUP -----
N = 100
state_space = np.arange(0, N-1, 1)
position_theta = 2*np.pi*(state_space/N)

posterior_pdf = np.zeros(N)
prior_pdf = np.zeros(N)

# ----- INITIALIZATION -----
# Initializing for maximum ignorance 
posterior_pdf = (1/N_NUM_STATES)*np.ones(N)
prior_pdf = (1/N_NUM_STATES)*np.ones(N)

updated_posterior_pdf = np.copy(posterior_pdf)
updated_prior_pdf = np.copy(prior_pdf)

initial_state = N//4

# ----- RECURSION -----
# Updating Prior PDF
for i in range(len(prior_pdf)):
    state_i_prob_value = 0.0
    for j in range(N):
        state_i_prob_value += calculate_process_model_PDF(state=i, process_noise_pdf=R_PROCESS_NOISE_PDF, num_states=N_NUM_STATES)*posterior_pdf[j]
    
    updated_posterior_pdf[i] = state_i_prob_value

print(updated_posterior_pdf)

