import numpy as np
import matplotlib.pyplot as plt

precision = 0
recall = 0

m1 = np.loadtxt("result_all1.txt")
final1 = np.zeros((m1.shape[0],2))

for i in range(0,255):
	for j in range(1, m1.shape[1]):
		if j%2 == 1:
			precision += m1[i,j]
		else:
			recall += m1[i,j]
	final1[i,0] = precision / 150
	final1[i,1] = recall / 150
	precision = 0
	recall = 0

m2 = np.loadtxt("result_all2.txt")
final2 = np.zeros((m2.shape[0],2))

for i in range(0,255):
	for j in range(1, m2.shape[1]):
		if j%2 == 1:
			precision += m2[i,j]
		else:
			recall += m2[i,j]
	final2[i,0] = precision / 150
	final2[i,1] = recall / 150
	precision = 0
	recall = 0

m3 = np.loadtxt("result_all.txt")
final3 = np.zeros((m3.shape[0],2))

for i in range(0,255):
	for j in range(1, m3.shape[1]):
		if j%2 == 1:
			precision += m3[i,j]
		else:
			recall += m3[i,j]
	final3[i,0] = precision / 150
	final3[i,1] = recall / 150
	precision = 0
	recall = 0


fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("Precision-Recall")
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')

x = final1[:,1]
y = final1[:,0]

ax1.plot(x,y, c='b', label='Method 1') 

x = final2[:,1]
y = final2[:,0]

ax1.plot(x,y, c='y', label='Method 2') 


x = final3[:,1]
y = final3[:,0]

ax1.plot(x,y, c='r', label='Our') 

leg = ax1.legend()

plt.show()
