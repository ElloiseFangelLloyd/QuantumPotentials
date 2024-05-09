import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helper_functions import potential, T, gauss

sns.set_palette(sns.husl_palette(10))

# DEFINE THE NUMBER OF POINTS TO BE 100
N = 100; 
# BOX LENGTH L
L = 5;

# DEFINE AN IDENTITY MATRIX OF SIZE N X N
Id = np.identity(N)

# DEFINE A SERIES OF X-VALUES, LINEARLY SPACED, FROM -L TO L, WITH N POINTS
xarr = np.linspace(-L,L,num=N)
#DEFINE DELTA X. AS THEY ARE LINEARLY SPACED, ANY X VALUES CAN BE USED
delx = xarr[2] - xarr[1]

# THE HARMONIC OSCILLATOR
# DEFINE A POTENTIAL MATRIX, CONSISTING OF THE IDENTITY MULTIPLIED BY THE POTENTIAL FORMULA
V = Id*potential(xarr)

# WITH THE CORRECT DIAGONALS
A = np.eye(N=N, M=N, k=1) + np.eye(N=N, M=N, k=-1) - 2*Id


# DEFINE THE HAMILTONIAN MATRIX
# THIS IS THE POTENTIAL MATRIX PLUS THE KINETIC ENERGY MATRIX
H = A*T(delx) + V

#FIND EIGENVALS AND EIGENVECS OF H
(eigenvalues, eigenvectors) = np.linalg.eig(H)

#EIGVALS AND EIGVECS ARE NOT SORTED BY DEFAULT
#SORT ACCORDING TO SORT_KEY, DEFINED BY THE EIGENVALS
sort_key = np.argsort(eigenvalues)
eigenvalues = eigenvalues[sort_key]
eigenvectors = eigenvectors[:, sort_key]

#HOW MANY STATES TO PLOT
n = 5;
#DEFINE POT SO THE POTENTIAL CAN BE PLOTTED
pot = potential(xarr)

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.plot(xarr, pot, linewidth=3)
plt.xlabel('X')
plt.ylabel('$\psi$')
plt.axis([-5, 5, -0.5, 5]) 
plt.title('Energy states of the harmonic oscillator in the potential V = 0.5 $x^2$')
plt.grid(linestyle='--')


#USE A LOOP TO PLOT
for i in range(n): 
    a = eigenvectors[:,i]*2+i #*2 IS SO THEY ARE ON BOTH SIDES, +i IS SO THEY ARE ELEVATED
    ax.plot(xarr, a, linewidth=3)
plt.savefig('harmonic_energy_states.png')


#DEFINE A NEW ARRAY WHICH IS THE ABSOLUTE VALUE OF THE EIGENVECTORS SQUARED, ELEMENTWISE
prob = np.array([np.abs(eigenvectors)**2])

#FIND THE INDEX OF THE LARGEST PROB VALUE
maxvalues = []
for i in range(10):
    B = np.argmax(prob[:,i])
    maxvalues.append(B)


#PLOTTING THE PROBABILITY DENSITY
plt.figure()
plt.plot(np.linspace(1,10,num=10), abs(xarr[maxvalues[0:30]]),'or')
plt.xlabel('n')
plt.ylabel('$x_n$')
plt.title('Position of the highest probability density of the $n$th eigenstate as a function of $n$')
plt.grid(linestyle='--')
plt.savefig('prob_density_with_n.png')


# Thresholds
X1, X2, X3 = 3, 5, 7

# Calculate boolean arrays
b1 = xarr > X1
b2 = xarr > X2
b3 = xarr > X3

# Sum probabilities
d1 = np.sum(prob[:, b1], axis=1)
d2 = np.sum(prob[:, b2], axis=1)
d3 = np.sum(prob[:, b3], axis=1)

# Transpose
dp1, dp2, dp3 = d1.transpose(), d2.transpose(), d3.transpose()

# Plotting
plt.figure()
plt.plot(np.linspace(1, 100, num=100), dp1, 'r', label=f'X > {X1}')
plt.plot(np.linspace(1, 100, num=100), dp2, 'b', label=f'X > {X2}')
plt.plot(np.linspace(1, 100, num=100), dp3, 'g', label=f'X > {X3}')
plt.xlabel('n')
plt.ylabel('$P_n$ (x > X)')
plt.xlim(0, 30)
plt.ylim(0, 0.4)
plt.title('Probability of a particle existing in a position x > X')
plt.grid(linestyle='--')
plt.savefig('probability.png')



V0 = 12.6
gpot= Id*gauss(xarr, V0)
H1 = A*T(delx) + gpot

(E, psi) = np.linalg.eig(H1)

sort_key_2 = np.argsort(E) #SORT, SAME AS LAST TIME
E = E[sort_key_2]
psi = psi[:, sort_key_2]

#DEFINE POTENTIAL FOR PLOTTING
pot2 = gauss(xarr, V0)

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.plot(xarr, pot2, linewidth=3)
plt.xlabel('X')
plt.ylabel('$\psi$')
plt.xlim(-5, 5)
plt.title('Energy states in the Gaussian potential')
plt.grid(linestyle='--')


#USE A LOOP TO PLOT
for i in range(n):
    b = psi[:,i]-(2*i) #USE -2i SO THE ENERGY STATES ARE NOT ON TOP OF EACH OTHER
    ax.plot(xarr, b, linewidth=3)
plt.savefig('gauss.png')
