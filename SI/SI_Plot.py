import numpy as np 
from matplotlib import pyplot as plt 
import pickle, glob

plt.style.use("ggplot")

#y = np.array([20,21,22,23])
#my_xticks = ['John','Arnold','Mavis','Matt']

#plt.plot(x, y)
layers = ['out_maxpool', 'out_conv2','out_conv3','out_conv4','out_featuremap','out_globalavg','out_fc'] 

"""
#CAMARKET:
si_all = {"age": [], "bags": [], "body": [], "body_colour": [], "body_type": [], "foot": [], "foot_colour": [],
            "gender": [], "head": [], "head_colour": [], "leg": [], "leg_colour": []}
"""
#CADUKE:
si_all = {"gender": [], "head": [], "head_colour": [],"cap": [], "cap_colour": [], "body": [], "body_colour": [], "bags": [],
            "umbrella": [], "face": [], "leg": [], "leg_colour": [],
            "foot": [], "foot_colour": [], "accessories": [], "position": [], "race": []}

for layer in layers:
    
    with open(layer+'.pkl', 'rb') as f:
        si = pickle.load(f)

    for key, value in si.items():
        si_all[key].append(value)

si_all.pop('accessories', None)

for key, value in si_all.items():
        plt.plot(value, label=key)

        for i in range(len(value)):
            plt.text(i, round(value[i], 2), '{:.2f}'.format(value[i]))

plt.legend(loc='best')
x = np.array([0,1,2,3,4,5,6])
plt.xticks(x, layers)
plt.show()
a=1