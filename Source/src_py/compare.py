import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import operator

def averagePrecisionRecall(m):
    final = np.zeros((m.shape[0],2))
    precision = 0.0
    recall = 0.0
    
    nimage = (m.shape[1] - 1) / 2

    for i in range(0,m.shape[0]):
        for j in range(1, m.shape[1]):
            if j%2 == 1:
                precision += m[i,j]
            else:
                recall += m[i,j]
                
        final[i,0] = precision / nimage
        final[i,1] = recall / nimage
        precision = 0.0
        recall = 0.0

    return final

deltax = 1.0 / 255.0

m1 = np.loadtxt("method1_result_all.txt")
final1 = averagePrecisionRecall(m1)

m2 = np.loadtxt("method2_result_all.txt")
final2 = averagePrecisionRecall(m2)

m3 = np.loadtxt("method_gt_result_all.txt")
final3 = averagePrecisionRecall(m3)

m4 = np.loadtxt("../results/result_all_mean_2.txt")
final4 = averagePrecisionRecall(m4)

m5 = np.loadtxt("../results/result_all_max_2.txt")
final5 = averagePrecisionRecall(m5)

m6 = np.loadtxt("../results/result_all_unique_2.txt")
final6 = averagePrecisionRecall(m6)

m7 = np.loadtxt("method_DL_result_all.txt")
final7 = averagePrecisionRecall(m7)

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("Precision-Recall")
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')

x = final1[:,1]
y = final1[:,0]

x_sorted, y_sorted = zip(*sorted(zip(x, y),key=operator.itemgetter(0), reverse=False))
area = np.trapz(y_sorted, x_sorted)
print("Method1 = {0}".format(area))
ax1.plot(x,y, c='red', label='Method 1')

x = final2[:,1]
y = final2[:,0]

x_sorted, y_sorted = zip(*sorted(zip(x, y),key=operator.itemgetter(0), reverse=False))
area = np.trapz(y_sorted, x_sorted)
print("Method2 = {0}".format(area))
ax1.plot(x,y, c='gold', label='Method 2') 

x = final3[:,1]
y = final3[:,0]

print(x)
print(y)

x_sorted, y_sorted = zip(*sorted(zip(x, y),key=operator.itemgetter(0), reverse=False))
area = np.trapz(y_sorted, x_sorted)
print("Ground Truth = {0}".format(area))
ax1.plot(x,y, c='chartreuse', label='Ground Truth')

x = final4[:,1]
y = final4[:,0]

x_sorted, y_sorted = zip(*sorted(zip(x, y),key=operator.itemgetter(0), reverse=False))
area = np.trapz(y_sorted, x_sorted)
print("Our method, Mean: {0}".format(area))

ax1.plot(x,y, c='blue', label='Mean')

x = final5[:,1]
y = final5[:,0]

x_sorted, y_sorted = zip(*sorted(zip(x, y),key=operator.itemgetter(0), reverse=False))
area = np.trapz(y_sorted, x_sorted)
print("Our method, Max: {0}".format(area))

ax1.plot(x,y, c='deeppink', label='Max')

x = final6[:,1]
y = final6[:,0]

x_sorted, y_sorted = zip(*sorted(zip(x, y),key=operator.itemgetter(0), reverse=False))
area = np.trapz(y_sorted, x_sorted)
print("Our method, Unique: {0}".format(area))

ax1.plot(x,y, c='y', label='Unique')

x = final7[:,1]
y = final7[:,0]

x_sorted, y_sorted = zip(*sorted(zip(x, y),key=operator.itemgetter(0), reverse=False))

area = np.trapz(y_sorted, x_sorted)
print("DL = {0}".format(area))
ax1.plot(x,y, c='darkmagenta', label='DeepSaliency')

ax1.set_yticks(np.arange(0.0, 1.0, 0.05))
ax1.set_xlim([0,1])
ax1.set_ylim([0,1]) 
leg = ax1.legend()
plt.grid()
plt.show()
