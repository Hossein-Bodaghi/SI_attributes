import numpy as np 
from matplotlib import pyplot as plt 

plt.style.use("ggplot")

clss = 'leg'

layers = np.load('./results/layers'+clss+'.npy')
trend = np.load('./results/trends'+clss+'.npy')

plt.plot(trend)

for i in range(len(trend)):
        plt.text(i, trend[i], '{:.2f}'.format(trend[i]))

plt.legend()
plt.title("Forward Selection")
plt.xlabel("Layers")
plt.ylabel("Separation Index")
plt.show()

