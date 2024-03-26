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
T_MAX_TIME = 100

P_OBJECT_MOVE_CCW = 0.5

# ----- HELPER FUNCTIONS -----
def calculate_process_noise(p_r):
    '''
    Calculates the process noise v(k-1). 

    Parameters:
        - p_r: Probability of returning 1 
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
    print(x_k)

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

def calculate_theta(state, num_states):
    '''
    Calculates the respective theta based on the state.

    Parameters:
        - state: State of the system 
        - num_states: Number of states
    '''
    return 2*np.pi*(state/num_states)

# Additional functions from solution code 
def update_prior(k, object_pdf):
    '''
    Predict the next state with given model 

    Parameters:
        - k (int): current time step 
    '''
    prior_pdf = np.zeros(N)

    for state in range(len(prior_pdf)):
        prior_pdf = (
            R_PROCESS_NOISE_PDF * object_pdf[k-1, np.mod(state-1, N_NUM_STATES)] + 
            (1 - R_PROCESS_NOISE_PDF) * object_pdf[k-1, np.mod(state+1, N_NUM_STATES)]
        )
    
    return prior_pdf

def calculate_dynamics(k, object_location) -> float:
    """
    Calculates the dynamics of the object and returns 
    the measured distance 

    Parameters:
        - k (int): Current time steps
        - object_location: Actual location of the object 
    """

    # Makes the object actually move with certain probability CCW or CW
    if np.random.rand() < P_OBJECT_MOVE_CCW:
        object_location[k, 0] = np.mod(object_location[k-1, 0] + 1, N_NUM_STATES)
    else:
        object_location[k, 0] = np.mod(object_location[k-1, 0] - 1, N_NUM_STATES)

    # Determine object location on unit circle 
    theta = calculate_theta(object_location[k,0])
    x_location, y_location = np.cos(theta), np.sin(theta)

    # Calculate actual distance to object
    real_distance = np.sqrt((L_SENSOR_DISTANCE - x_location)**2 + y_location**2)
    measured_distance = real_distance + calculate_sensor_noise(E_SENSOR_NOISE_TOLERANCE)

    return measured_distance


def update_measurement(k, prior_pdf, measured_distance, object_pdf):
    """
    Combines prediction with measurement 

    Parameters:
        - k (int): Current time step 
        - prior_pdf [np.ndarray, shape:<1, N>]: Prior distribution 
        - measured_distance [float]: Distance reported by the sensor 
    """
    for i in range(N_NUM_STATES):
        theta =  calculate_theta(i)
        x_location_hypothesis = np.cos(theta)
        y_location_hypothesis = np.sin(theta)
        distance_hypothesis = np.sqrt((L_SENSOR_DISTANCE - x_location_hypothesis)**2 + y_location_hypothesis**2)

        # This is where the key fusion is included (?)
        obs_meas_cond_prob = 0.0
        if np.abs(measured_distance - distance_hypothesis) < E_SENSOR_NOISE_TOLERANCE:
            obs_meas_cond_prob = 1 / (2*E_SENSOR_NOISE_TOLERANCE)
        else:
            obs_meas_cond_prob = 0.0

        object_pdf[k, i] = obs_meas_cond_prob * prior_pdf[0, i] # Why do we combine both?

# ----- INITIALIZATION -----
N = 100
state_space = np.arange(0, N, 1)
posterior_pdf = np.zeros(N)

# Keeps track of the probability that the object is at a 
# location i at time t
object_pdf = np.zeros((T_MAX_TIME + 1, N_NUM_STATES))
object_pdf[0,:] = 1/N_NUM_STATES * np.ones(N_NUM_STATES)    # Setting to 1/N to indicate all positions are likely    

object_location = np.zeros((T_MAX_TIME, 1))
object_location[0, 0] = N//4

# ----- RECURSION -----
for k in range(1, T_MAX_TIME, 1):

    measured_distance = calculate_dynamics(k, object_location)
    prior = update_prior(k)

# # Updating posterior PDF 
# for i in range(len(posterior_pdf)):
#     theta_i = calculate_theta(state_space[i], N_NUM_STATES)
#     measurement_model_value = calculate_sensor_model_PDF(E_SENSOR_NOISE_TOLERANCE, L_SENSOR_DISTANCE, theta_i)
#     prior_value = updated_prior_pdf[j]

#     normalization_value = 0.0
#     for j in range(N):
#         theta_j = calculate_theta(state_space[j], N_NUM_STATES)
#         normalization_value += calculate_sensor_model_PDF(E_SENSOR_NOISE_TOLERANCE, L_SENSOR_DISTANCE, theta_j)*updated_prior_pdf[j]

#     updated_posterior_pdf[i] = (measurement_model_value*prior_value)/normalization_value

# print("Updated posterior PDF: ", updated_posterior_pdf)
# print("Sum check: ", np.sum(updated_posterior_pdf))
# print("")