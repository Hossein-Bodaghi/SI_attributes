import numpy as np
import matplotlib.pyplot as plt

a = np.load("./Desktop/si_ID_8.01.npy")
b = np.load(./SI_Results_random.npy")
plt.style.use("ggplot")
plt.plot(layers, a, label="V8.01")
plt.plot(layers, b, label="Pretrain")
for i in range(len(a)):
        plt.text(i, a[i], '{:.3f}'.format(a[i]))
        plt.text(i, b[i], '{:.3f}'.format(b[i]))
"""
si = []
#si = np.append(si,np.load("C:/Users/ASUS/Desktop/si_10k_max.npy"))
si = np.append(si,np.load("C:/Users/ASUS/Desktop/si_body_10k.npy"))
si = np.reshape(si, (-1, 3))

#si = np.round(si, 2)

plt.style.use("ggplot")

shirt=si[:,0]
coat = si[:,1]
top = si[:,2]
plt.plot(layers, shirt, label="shirt")
plt.plot(layers, coat, label="coat")
plt.plot(layers, top, label="top")
for i in range(len(si)):
        plt.text(i, shirt[i]+0.02, '{:.3f}'.format(shirt[i]))
        plt.text(i, coat[i]+0.02, '{:.3f}'.format(coat[i]))
        plt.text(i, top[i]+0.02, '{:.3f}'.format(top[i]))

"""
"""
#classes
bags=si[:,0]
body = si[:,1]
body_colour = si[:,2]
foot = si[:,3]
foot_colour = si[:,4]
leg = si[:,5]
leg_colour = si[:,6]
head = si[:,7]
gender = si[:,8]
body_type = si[:,9]
plt.plot(layers, bags, label="bags")
plt.plot(layers, body, label="body")
plt.plot(layers, body_colour, label="body_colour")
plt.plot(layers, foot, label="foot")
plt.plot(layers, foot_colour, label="foot_colour")
plt.plot(layers, leg, label="leg")
plt.plot(layers, leg_colour, label="leg_colour")
plt.plot(layers, head, label="head")
plt.plot(layers, gender, label="gender")
plt.plot(layers, body_type, label="body_type")
for i in range(len(si)):
        plt.text(i, bags[i], '{:.2f}'.format(bags[i]))
        plt.text(i, body[i], '{:.2f}'.format(body[i]))
        plt.text(i, body_colour[i], '{:.2f}'.format(body_colour[i]))
        plt.text(i, foot[i], '{:.2f}'.format(foot[i]))
        plt.text(i, foot_colour[i], '{:.2f}'.format(foot_colour[i]))
        plt.text(i, leg[i], '{:.2f}'.format(leg[i]))
        plt.text(i, leg_colour[i], '{:.2f}'.format(leg_colour[i]))
        plt.text(i, head[i], '{:.2f}'.format(head[i]))
        plt.text(i, gender[i], '{:.2f}'.format(gender[i]))
        plt.text(i, body_type[i], '{:.2f}'.format(body_type[i]))"

"""

plt.legend()
plt.xlabel("Layers")
plt.title("Separation Index")
plt.show()

print('a')