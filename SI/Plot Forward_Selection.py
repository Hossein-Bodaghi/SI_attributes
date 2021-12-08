import numpy as np 
from matplotlib import pyplot as plt 
from functions import layers_num_corrector


plt.style.use("ggplot")

clss = 'leg'

layers = np.load('./results/layers'+clss+'.npy')
trend = np.load('./results/trends'+clss+'.npy')


x1, y1 = [0, len(trend)], [0.984, 0.984]
plt.text(0, 0.975, '{:.3f}'.format(0.975))
plt.plot(x1,y1)
final = layers_num_corrector(layers)

plt.plot(trend)

for i in range(len(trend)):
        if i%5 == 0:
                plt.text(i, trend[i], '{:.3f}'.format(trend[i]))

plt.legend()
plt.title("Forward Selection")
plt.xlabel("Layers")
plt.ylabel("Separation Index")
plt.show()

