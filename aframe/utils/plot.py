import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)


def cylinder(radius, ua, ub):

    L = np.linalg.norm(ub - ua)
    v = (ub - ua)/L


    not_v = np.array([1, 0, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])

    # make a unit vector perpendicular to v:
    n = np.cross(v, not_v)/np.linalg.norm(np.cross(v, not_v))

    # make a unit vector perpendicular to v and n
    m = np.cross(v, n)


    t = np.linspace(0, L, 100)
    theta = np.linspace(0, 2*np.pi, 100)
    t, theta = np.meshgrid(t, theta)

    X, Y, Z = [ua[i] + v[i]*t + radius*np.sin(theta)*n[i] + radius*np.cos(theta)*m[i] for i in [0, 1, 2]]
    
    return X, Y, Z






if __name__ == '__main__':

    ua = np.array([1,3,2])
    ub = np.array([8,5,9])
    r = 3

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    X, Y, Z = cylinder(r, ua, ub)


    ax.plot_surface(X,Y,Z)
    plt.show()