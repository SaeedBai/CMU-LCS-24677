import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# 1.1 - Check controllability and observability

# Vehicle dynamics parameters
lr = 1.39
lf = 1.55
Ca = 20000
Iz = 25854
m = 1888.6
g = 9.81
f = 1

V = [2,5,8]

for idx, val in enumerate(V):
    
    xdot = val
    
    # State-space equation
    A = np.array([[0,1,0,0],[0,-4*Ca/(m*xdot),4*Ca/m,2*Ca*(lr-lf)/(m*xdot)] \
                  ,[0,0,0,1],[0,(2*Ca)*(lr-lf)/(Iz*xdot),(2*Ca)*(lf-lr)/Iz, \
                   (-2*Ca)*(lf**2 + lr**2)/(Iz*xdot)]])
    B = np.array([[0],[2*Ca/m],[0],[2*Ca*lf/Iz]])
    # phidot_des term is ignored for B
    C = np.identity(4)
    D = np.zeros((4,1))
    
    # Check controllability by manually building P
    P1 = B
    P2 = np.dot(A,B)
    P3 = np.dot(np.linalg.matrix_power(A,2),B)
    P4 = np.dot(np.linalg.matrix_power(A,3),B)
    
    # Check observability by manually building Q
    Q1 = C
    Q2 = np.dot(C,A)
    Q3 = np.dot(C,np.linalg.matrix_power(A,2))
    Q4 = np.dot(C,np.linalg.matrix_power(A,3))
    
    P = np.concatenate((P1,P2,P3,P4),axis=1)
    Q = np.vstack((Q1,Q2,Q3,Q4))
    
    # Determine the rank of both P and Q
    print('P for {} m/s has rank {}'.format(xdot,np.linalg.matrix_rank(P)))
    print('Q for {} m/s has rank {}'.format(xdot,np.linalg.matrix_rank(Q)))

print("Thus the system is controllable and observable for every value of Vx tested")
print(" ")

# 1.2 - Graphs of singular value ratios and poles
    
V = np.linspace(1, 40, 40)
logsigma_arr = []
pole1 = []
pole2 = []
pole3 = []
pole4 = []

for idx, val in enumerate(V):
    
    xdot = val
    
    # State-space equation
    A = np.array([[0,1,0,0],[0,-4*Ca/(m*xdot),4*Ca/m,2*Ca*(lr-lf)/(m*xdot)] \
                  ,[0,0,0,1],[0,(2*Ca)*(lr-lf)/(Iz*xdot),(2*Ca)*(lf-lr)/Iz, \
                   (-2*Ca)*(lf**2 + lr**2)/(Iz*xdot)]])
    B = np.array([[0],[2*Ca/m],[0],[2*Ca*lf/Iz]])
    C = np.identity(4)
    D = np.zeros((4,1))
    
    # Manually build P
    P1 = B
    P2 = np.dot(A,B)
    P3 = np.dot(np.linalg.matrix_power(A,2),B)
    P4 = np.dot(np.linalg.matrix_power(A,3),B)
    P = np.concatenate((P1,P2,P3,P4),axis=1)

    # Get first and last singular values of P 
    [u,sigma,v] = np.linalg.svd(P)
    sigma_1 = sigma[0]
    sigma_n = sigma[-1]

    # Determine logarithm of their ratio
    logsigma = np.log10(sigma_1/sigma_n)
    logsigma_arr = np.append(logsigma_arr,logsigma)
    
    # Get eigenvalues of A, which are the poles of the system
    [lam,v] = np.linalg.eig(A)
    # Only get the real components of each pole for plotting
    realroots = lam.real
    pole1 = np.append(pole1,realroots[0])
    pole2 = np.append(pole2,realroots[1])
    pole3 = np.append(pole3,realroots[2])
    pole4 = np.append(pole4,realroots[3])

plt.title('log$_{10}(\sigma_1/\sigma_n)$ vs. V')
plt.xlabel('V (m/s)')
plt.ylabel('log$_{10}(\sigma_1/\sigma_n)$')
plt.plot(V, logsigma_arr)


fig = plt.figure()
    
plt.subplot(221)
plt.title('Re($p_1$)')
plt.plot(V, pole1)

plt.subplot(222)
plt.title('Re($p_2$)')
plt.plot(V, pole2)

plt.subplot(223)
plt.title('Re($p_3$)')
plt.plot(V, pole3)

plt.subplot(224)
plt.title('Re($p_4$)')
plt.plot(V, pole4)

fig.tight_layout()
plt.show()

print("Part A Explanation: The ratio of singular values of the controllability matrix reflects the defectiveness of the system. The defectiveness of the system is inversely proportional to the ratio, i.e. the smaller the ratio, the less the system is likely to be defective. Therefore, the system is comparatively more controllable in the lateral direction at higher longitudinal velocities. This fits with the intuition that it is easier to make steering changes in a car when the car is traveling at a faster speed. If the car is traveling at a very low speed, it is much more difficult to control the car in the lateral direction.")
print(" ")
print("Part B Explanation: The system is a second-order system with two poles and two zeros. With an increase in the longitudinal velocity, the conjugate pole pairs move closer to the imaginary axis, indicating that the system tends to be less stable as the velocity increases. Notably, one of the poles goes above the imaginary axis at a value of x_dot = 34 m/s - in other words, the system becomes unstable at this velocity.")
print(" ")
print(" In conclusion, as the longitudinal velocity increases, it is easier to make a change in the lateral state (controllability) though there is a higher risk of the system becoming unstable. When l_{r}C_{a}<l_{f}C_{a}, as the longitudinal velocity increases, less control input is needed to steer the car. This is true until the car reaches a velocity (denoted as the critical velocity), where even with zero steering angle, the car steers in a direction. For our system, this critical velocity appears to be at x_dot = 34 m/s. If the velocity continues to increase, a left steering input is required to turn to the right. This phenomenon is called oversteering.")