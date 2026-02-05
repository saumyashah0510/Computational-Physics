import numpy as np

def gen_euler(state,dt,derivative_func,**args):

    """
    A general purpose Euler method step.
    
    :param state: array-like, current values of variables (e.g., [Na, Nb] or [x, y, vx, vy])
    :param dt: float, time step
    :param derivative_func: function that calculates [d(var1)/dt, d(var2)/dt, ...]
    :param args: any extra parameters needed for the physics (like tau or k)

    Returns:
    next_state: numpy array of updated values

    """

    state = np.array(state)

    derivatives = np.array(derivative_func(state,**args))

    next_state = state + derivatives*dt
    return next_state

