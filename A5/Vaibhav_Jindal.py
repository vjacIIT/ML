import numpy as np
import pandas as pd
import csv
from statistics import mean
from statistics import mode
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

df = pd.read_csv('Patients.csv', header=None)
r, c = np.shape(df)

X = np.zeros(shape=(r,c), dtype=np.float64)

# freq_list stores frobenius norm of each input vector
freq_list = []
for i in range (r):
	X[i] = df.iloc[i,:]
	freq_list.append(int(np.linalg.norm(X[i])))

print(X[0])
# Same frobenius points means that points are very near (assumption)
#print(freq_list)
freq_no = mode(freq_list)				# number with maximum frequency
freq = freq_list.count(freq_no)
print("Min no of samples required for core point\n",4*freq)


# Distance (Frobenius norm) between any pair of input dataset
eps_list = []
for i in range(r):
	for j in range(i+1,r):
		eps_list.append(int(np.linalg.norm(X[i]-X[j])))

# Epsilon (radius) is mean of all the distances
eps = mean(eps_list)
print("\nEpsilon (radius) size\n",eps)

# DBSCAN
db = DBSCAN(eps=eps, min_samples=4*freq).fit(X)
labels = db.labels_

#data = []
#for i in range(r):
	#if(labels[i]==-1):
		#data.append(eps[i])

#data.sort()
#print(data)

file = open('Vaibhav_Jindal.csv','w')
writer = csv.writer(file)
writer.writerows(map(lambda x:[abs(x)], labels))
file.close()

n_noise = list(labels).count(-1)
print('\nNo of noise points\n',n_noise)
