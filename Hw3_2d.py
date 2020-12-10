import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from filterpy.kalman import KalmanFilter

alpha_0 = np.random.normal(scale =10,size = (1))
alpha_1 = np.random.normal(scale = 10, size = (1))
f = KalmanFilter (dim_x=2, dim_z=1)
X = np.array([alpha_0, alpha_1])
f.F = np.array([[1,0],[0,1]])

forward_prediction = []
state_estmation = []


for i in range(0,100):
    f.H = np.array([[1.,i]])
    f.P *= (1/20 +f.H.T@f.H)**-1
    noise = np.random.normal(scale=1, size=(1, 1))
    f.R = noise
    observation = np.array(X[0] + X[1] * i +noise)
    f.predict()
    f.update(observation)
    state_estmation.append(f.x)



# forward_prediction = [0]*100
# state_update = [0]*100
# print("here")
# X_hat_0 = [0,0]
#
# predicted_error = []
# forward_prediction[0] = X_hat_0
# for i in range(1, 100):
#
#
#     H_i = [1, i]
#
#     error = np.array(forward_prediction[i-1]) - np.array(X)
#     state_update[i] = forward_prediction[i-1]+ error
#     innovation = (observation-np.array(H_i)@np.array(forward_prediction[i-1]))[0]
#     print(innovation)
#     R_ei = autocorrelation(innovation)
#     state_estimate = forward_prediction[i-1] + P_i*np.array((innovation)/(R_ei))
#     forward_estimate = state_estimate
#     state_update[i-1] = state_estimate
#     forward_prediction[i] = forward_estimate

print("done plotting")
X_extended = []

for _ in range(len(state_estmation)):
    X_extended.append(X)



mse_a0 = []
mse_a1 = []
predicted_error = []
P_0= 1/(10+10)
for i in range(len(state_estmation)):
    mse_a0.append(mean_squared_error(state_estmation[i][0], X[0]))
    mse_a1.append(mean_squared_error(state_estmation[i][1], X[1]))
    P_i = (P_0 + 2 * i) ** -1
    predicted_error.append(P_i)


plt.scatter(np.arange(0, len(mse_a0)), mse_a0, label = "Actual error a0")
plt.scatter(np.arange(0, len(mse_a1)), mse_a1, label = "Actual error a1")
plt.scatter(np.arange(0, len(mse_a0)), predicted_error, label = "Predicted error")
plt.legend()
plt.show()









