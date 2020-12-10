import numpy as np
from matplotlib import pyplot as plt


"""SGD with poor approximation of mu"""

theta_0 = 30
time = np.arange(1,100)
steps = 0.1/time
time = list(time)



curr_theta_t = [theta_0]
curr_theta = theta_0
for t in time:
    curr_theta = curr_theta - steps[t-1]*2*curr_theta
    curr_theta_t.append(curr_theta)


time.insert(0,0)
plt.plot(time, np.array(curr_theta_t)**2, label = "SGD")


"SCSG with same setting"
b = 1
m = 1
step_size = 0.1
theta_0 = 30
time = list(np.arange(1,100))
all_theta_t = []
for _ in range(100):
    theta_t = [theta_0]
    curr_theta  =  theta_0
    for t in time:
        v_j = 2*curr_theta
        N_j = np.random.geometric(1/2)
        theta_j_0 = curr_theta
        curr_theta_j = theta_j_0
        for k in range(N_j):
            v_hat_k_minus_one = 2*curr_theta_j - 2*theta_j_0+v_j
            curr_theta_j =curr_theta_j-step_size*v_hat_k_minus_one

        curr_theta = curr_theta_j
        theta_t.append((curr_theta))
    all_theta_t.append(theta_t)


time.insert(0,0)

plt.plot(time, np.mean(np.array(all_theta_t)**2, axis=0), label = " Average SCSG")
plt.title("Converges of SGD vs. SCSG in simple setting")
plt.ylabel("Theta^2")
plt.xlabel("Time")
plt.legend()

plt.show()



"LMS vs SGD"

theta_0 = 30
time = np.arange(1,100)
step = 1
time = list(time)



curr_theta_t = [theta_0]
mean_theta = [theta_0]
curr_theta = theta_0
for t in time:
    curr_theta = curr_theta - step*2*curr_theta
    curr_theta_t.append(curr_theta)
    mean_theta.append(np.mean(curr_theta_t))


time.insert(0,0)

plt.plot(time, np.array(curr_theta_t)**2, label = "Theta squared")
plt.plot(time, np.array(curr_theta_t), label = "Theta")
plt.plot(time, np.array(mean_theta)**2, label = " Mean Theta squared")
plt.plot(time, np.array(mean_theta), label = " Mean Theta")
plt.xlabel("Time")
plt.title("Convergence of Theta vs. Mean Theta in SGD")
plt.legend()
plt.show()








