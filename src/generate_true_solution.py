import numpy as np

dt = 1e-3 
T = 1.0 
omega = 0.271 
alpha = 0.871

t = np.arange(0, T + dt, dt) 
x = alpha * np.cos(omega * t)

with open('true_solution.dat', 'w+') as f: 
    f.write("{}, {}\n".format(dt, len(t))) 
    for v in x: 
        f.write("{}\n".format(v))
