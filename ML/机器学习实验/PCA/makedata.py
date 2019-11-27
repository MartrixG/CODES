import numpy as np

def makedata(n, miu, sigma, theta):
    a = 1
    b = 0.2
    th=np.linspace(0, 10, int(n / 10))
    x = np.expand_dims(np.tile(a * np.exp(b * th) * np.cos(th), 10) + np.random.normal(miu, sigma, (n)), axis=0)
    y = np.expand_dims(np.tile(a * np.exp(b * th) * np.sin(th), 10) + np.random.normal(miu, sigma, (n)), axis=0)
    z = np.expand_dims(np.array(np.linspace(-1.5, 1.5, n) + np.random.normal(miu, sigma, (n))), axis=0)
    mo = np.concatenate([np.concatenate([np.concatenate([x, y], axis=0), z], axis=0), np.ones((1, n))], axis=0)
    ro_x = np.zeros((4, 4))
    ro_x[0][0] = 1.0
    ro_x[1][1] = np.cos(theta)
    ro_x[1][2] = -np.sin(theta)
    ro_x[2][1] = np.sin(theta)
    ro_x[2][2] = np.cos(theta)
    ro_x[3][3] = 1.0
    ro_y = np.zeros((4, 4))
    ro_y[0][0] = np.cos(theta)
    ro_y[1][1] = 1
    ro_y[0][2] = -np.sin(theta)
    ro_y[2][0] = np.sin(theta)
    ro_y[2][2] = np.cos(theta)
    ro_y[3][3] = 1.0
    mo = np.dot(ro_x, mo)
    mo = np.dot(ro_y, mo)
    return np.delete(mo, -1, axis=0).T