import numpy as np

def backward_euler(f, y0, t, dt):
    """
    Solves the ODE y' = f(t, y) using the backward Euler method.

    Parameters:
    f : function
        The function that defines the ODE, f(t, y).
    y0 : float or np.array
        The initial condition at t[0].
    t : np.array
        The array of time points where the solution is computed.
    dt : float
        The time step.

    Returns:
    y : np.array
        The array containing the solution at each time point.
    """
    n = len(t)
    y = np.zeros((n,) + np.shape(y0))
    y[0] = y0

    for i in range(1, n):
        # Use the previous value of y as the initial guess
        y_guess = y[i-1]

        # Fixed-point iteration to solve the implicit equation
        for _ in range(10):  # You can adjust the number of iterations
            y_new = y[i-1] + dt * f(t[i], y_guess)
            y_guess = y_new

        y[i] = y_new

    return y

# Example usage:
def f(t, y):
    return -y  # Example: dy/dt = -y

y0 = 1.0
t = np.linspace(0, 5, 100)
dt = t[1] - t[0]

y = backward_euler(f, y0, t, dt)

# Now y contains the solution to the ODE at each time point in t