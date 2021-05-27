import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys

#############################################
# Parameters ################################
#############################################

# Default
H = 15
V = 15
iterations = 100
P = 0.3
P_noise_pix = 0.2
exact = True
if len(sys.argv) > 1:
    # From console
    H = int(sys.argv[1])
    V = int(sys.argv[2])
    iterations = int(sys.argv[3])
    P = float(sys.argv[4])
    P_noise_pix = float(sys.argv[5])
    exact = bool(sys.argv[6])

#############################################
# Image generation ##########################
#############################################

img = np.zeros((V, H))

row_numb = []
column_numb = []

# Generate rows and colums
for i in range(V):
    rand = random.random()
    if rand < P:
        img[i:i+1, :] = 1
        row_numb.append(i)

for i in range(H):
    rand = random.random()
    if rand < P:
        img[:, i:i+1] = 1
        column_numb.append(i)  

fig, ax = plt.subplots(1, 3 + int(exact))
ax[0].imshow(img)

# Noise the image
for i in range(V):
    for j in range(H):
        rand = random.random()
        if rand < P_noise_pix:
            img[i][j] = 1 - img[i][j]

ax[1].imshow(img)

#############################################
# Gibbs sampler #############################
#############################################

start_time = time.time()
r = np.zeros((V))
c = np.zeros((H))

# Functions for calculating probabilities
for i in range(V):
    rand = random.random()
    if rand < P:
        r[i] = 1

def p_I_r_c(im,r, c):
    if (im == 0):
        if ((r == 1) or (c == 1)):
            return P_noise_pix
        else:
            return 1 - P_noise_pix
    elif (im == 1):
        if ((r == 1) or (c == 1)):
            return 1 - P_noise_pix
        else:
            return P_noise_pix

def g_j(c, r):
    p = 1
    g_j = ((P**c)*((1-P)**(1-c)))
    for i in range(V):
        p = p* p_I_r_c(img[i][j],r[i], c)
    g_j = g_j*p
    return g_j

def g_i(r, c):
    p = 1
    g_i = ((P**r)*((1-P)**(1-r)))
    for j in range(H):
        p = p* p_I_r_c(img[i][j],r, c[j])
    g_i = g_i*p
    return g_i

# Sampling
for k in range(iterations):
    for j in range(H):
        rand = random.random()
        if rand < g_j(1, r)/(g_j(0, r)+g_j(1, r)):
            c[j] = 1
        else:
            c[j] = 0
    
    for i in range(V):
        rand = random.random()
        if rand < g_i(1, c)/(g_i(0, c)+g_i(1, c)):
            r[i] = 1
        else:
            r[i] = 0

img_result = np.zeros((V, H))

# Saving the result
for i in range(V):
    if r[i] == 1:
        img_result[i:i+1, :] = 1
        
for j in range(H):
    if c[j] == 1:
        img_result[:, j:j+1] = 1

ax[2].imshow(img_result)

fig.set_figwidth(12)    
fig.set_figheight(6)

print("Gibbs sampler time taken - ", (time.time() - start_time))
if not exact:
    plt.show()

#############################################
# Exact solution ############################
#############################################

if exact:
    start_time = time.time()
    probs = []

    def get_bits(i, size):
        bits = np.zeros(size, int)
        cur = i
        for j in range(size):
            bits[size-1-j] = cur % 2
            cur /= 2
        return bits
    
    # Function for calculating probabilities
    def p_c_I(c):
        p_c = 1
        for j in range(H):
            p_c *= (((P**c[j])*((1-P)**(1-c[j]))))
        p = 1
        for i in range(V):
            s = 0
            for r in range(2):
                p_I_c_r = 1
                for j in range(H):
                    p_I_c_r *= p_I_r_c(img[i][j], r, c[j])
                s+= (((P**r)*((1-P)**(1-r)))) * p_I_c_r
            p *= s
   
        return p_c * p  
    
    # Iterating through all possible configurations of columns
    for i in range(2**H):
        bits = get_bits(i, H)
        probs.append(p_c_I(bits))

    # Normalizing
    probs = np.array(probs)
    probs = probs / np.sum(probs)
    
    rand = random.random()

    k = -1
    w = 0.
    # Generating rows
    for i in range(2**H):
        w = np.sum(probs[0:i + 1])
        if (rand < w):
            k = i
            break

    # Getting configuration of generated rows from index
    c = get_bits(k, H)

    # Generating rows based on generated columns
    for i in range(V):
            rand = random.random()
            if rand < g_i(1, c)/(g_i(0, c)+g_i(1, c)):
                r[i] = 1
            else:
                r[i] = 0

    # Saving the result
    img_result_2 = np.zeros((V, H))

    for i in range(V):
        if r[i] == 1:
            img_result_2[i:i+1, :] = 1
            
    for j in range(H):
        if c[j] == 1:
            img_result_2[:, j:j+1] = 1
    ax[3].imshow(img_result_2)
    print("Exact solution time taken - ", (time.time() - start_time))
    plt.show()

    








