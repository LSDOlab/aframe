import numpy as np

def midpoint_rule(f, y0, t, dt):
    """
    Solves the ODE y' = f(t, y) using the midpoint rule.

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
        # Compute the midpoint
        t_mid = t[i-1] + dt/2
        y_mid = y[i-1] + (dt/2) * f(t[i-1], y[i-1])

        # Compute the next value using the midpoint
        y[i] = y[i-1] + dt * f(t_mid, y_mid)

    return y

# Example usage:
def f(t, y):
    return -y  # Example: dy/dt = -y

y0 = 1.0
t = np.linspace(0, 5, 100)
dt = t[1] - t[0]

y = midpoint_rule(f, y0, t, dt)

# Now y contains the solution to the ODE at each time point in t
