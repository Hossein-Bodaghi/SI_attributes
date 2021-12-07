import numpy as np 
from matplotlib import pyplot as plt 

plt.style.use("ggplot")

layers =np.load('./results/layerindexleg.npy')
x = np.arange(1,1+layers.shape[0]) 
y = np.load('./results/trendbody.npy') 
plt.title("Forward Selection") 
plt.xlabel("Number of selected layers") 
plt.ylabel("SI") 
plt.plot(x,y)
for i in range(len(x)):
        plt.text(i, x[i], '{:.2f}'.format(x[i]))
        plt.text(i, y[i], '{:.2f}'.format(y[i])) 
plt.legend()
plt.xlabel("Layers")
plt.ylabel("Separation Index")
#plt.show()
to_add = np.zeros((26,))
def layers_number_decoder (layers):
      for i in range(layers.shape[0]):
        for j in range(i):
            if layers[j]<=layers[i]:
                to_add[i]+=1
      return to_add
to_add = layers_number_decoder(layers)
final = to_add + layers
np.save('./corrected/bodylayers_decoded.npy', final)