import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


def averagePrecisionRecall(m):
	final = np.zeros((m.shape[0],2))
	precision = 0
	recall = 0

	for i in range(0,255):
		for j in range(1, m.shape[1]):
			if j%2 == 1:
				precision += m[i,j]
			else:
				recall += m[i,j]
		final[i,0] = precision / 1000
		final[i,1] = recall / 1000
		precision = 0
		recall = 0

	return final


#m1 = np.loadtxt("result_all_mean.txt")
#final1 = averagePrecisionRecall(m1)

#m2 = np.loadtxt("result_all_max.txt")
#final2 = averagePrecisionRecall(m2)

#m3 = np.loadtxt("result_all_unique.txt")
#final3 = averagePrecisionRecall(m3)

m4 = np.loadtxt("result_all_mean_2.txt")
final4 = averagePrecisionRecall(m4)

m5 = np.loadtxt("result_all_max_2.txt")
final5 = averagePrecisionRecall(m5)

m6 = np.loadtxt("result_all_unique_2.txt")
final6 = averagePrecisionRecall(m6)

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("Precision-Recall")
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')

#x = final1[1:,1]
#y = final1[1:,0]

#ax1.plot(x,y, c='b', label='Mean') 

#x = final2[1:,1]
#y = final2[1:,0]

#ax1.plot(x,y, c='y', label='Max') 

#x = final3[1:,1]
#y = final3[1:,0]

#ax1.plot(x,y, c='r', label='Unique')

x = final4[1:,1]
y = final4[1:,0]

area = np.trapz(y, dx=0.001)
print(area)
area = simps(y, dx=0.001)
print(area)

ax1.plot(x,y, c='r', label='Mean')

x = final5[1:,1]
y = final5[1:,0]

area = np.trapz(y, dx=0.001)
print(area)
area = simps(y, dx=0.001)
print(area)

ax1.plot(x,y, c='g', label='Max')

x = final6[1:,1]
y = final6[1:,0]

area = np.trapz(y, dx=0.001)
print(area)
area = simps(y, dx=0.001)
print(area)

ax1.plot(x,y, c='b', label='Unique')

ax1.set_xlim([0,1])
ax1.set_ylim([0,1]) 
leg = ax1.legend()

plt.show()
