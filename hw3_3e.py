import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from filterpy.kalman import KalmanFilter, FixedLagSmoother


f = KalmanFilter (dim_x=1, dim_z=1)
f.F = np.array([-0.5])
f.H = np.array([1])

X = [0]
state_estimate = []
covariances = []
Q_s = [0]*5000
R_s = [0]*5000
noise_s = [0]*5000

for i in range(1,5000):
    noise = np.random.normal(scale=1, size=(1, 1))
    f.R = noise
    f.Q = np.random.normal(scale=1, size=(1, 1))
    Q_s[i] = f.Q
    R_s[i] = f.R
    observation = -0.5*X[i-1] + f.Q + noise
    noise_s[i] = noise
    X.append((-0.5*X[i-1] + f.Q)[0])
    f.predict()
    f.update(observation)
    state_estimate.append(f.x)
    covariances.append(f.P)


fls = FixedLagSmoother(dim_x=2, dim_z=1, N = 100)
f.F = np.array([-0.5])
f.H = np.array([1])

for i in range(1,5000):
    f.R = R_s[i]
    f.Q = Q_s[i]
    observation = X[i]+noise_s[i]
    fls.smooth(observation)



mse_forward = []
mse_smoothed = []

for i in range(len(state_estimate)):
    mse_forward.append(mean_squared_error([X[i]], [state_estimate[i]]))
    mse_smoothed.append(mean_squared_error([X[i]], fls.xSmooth[i][0]))

plt.scatter(np.arange(0, len(mse_forward)), mse_forward, label = "Forward Kalman MSE")
plt.scatter(np.arange(0, len(mse_smoothed)), mse_smoothed, label = "Smoothed Kalman MSE")
plt.legend()
plt.show()






