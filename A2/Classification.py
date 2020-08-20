from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA


fig = plt.figure()
ax = fig.gca(projection='3d')

# draw sphere

radius1 = 1
radius2 = 2
radius3 = 3
centre = [1,1,1]


pts = 10
u = np.linspace(0, 2 * np.pi, pts)
noise = np.random.normal(0,0.25,u.shape)
u = u + noise
v = np.linspace(0, np.pi, pts)
noise = np.random.normal(0,0.25,v.shape)
v = v + noise
n_sp1 = pts*pts												# npoints for sphere 1 = 100
X1 = centre[0] + radius1 * np.outer(np.cos(u), np.sin(v))
Y1 = centre[1] + radius1 * np.outer(np.sin(u), np.sin(v))
Z1 = centre[2] + radius1 * np.outer(np.ones(np.size(u)), np.cos(v))
X1 = X1.reshape((n_sp1,1))
Y1 = Y1.reshape((n_sp1,1))
Z1 = Z1.reshape((n_sp1,1))
ax.scatter3D(X1,Y1,Z1,c='#FF0000');								# Red color
class1 = np.c_[X1,Y1,Z1]											# class1 points, dim=pts*3
#print(np.shape(class1))


pts = 20														# npoints for sphere 2 = 400
u = np.linspace(0, 2 * np.pi, pts)
noise = np.random.normal(0,0.25,u.shape)
u = u + noise
v = np.linspace(0, np.pi, pts)
noise = np.random.normal(0,0.25,v.shape)
v = v + noise
n_sp2 = pts*pts
X2 = centre[0] + radius2 * np.outer(np.cos(u), np.sin(v))
Y2 = centre[1] + radius2 * np.outer(np.sin(u), np.sin(v))
Z2 = centre[2] + radius2 * np.outer(np.ones(np.size(u)), np.cos(v))
X2 = X2.reshape((n_sp2,1))
Y2 = Y2.reshape((n_sp2,1))
Z2 = Z2.reshape((n_sp2,1))
ax.scatter3D(X2,Y2,Z2,c='#8500FF');								# Blue
class2 = np.c_[X2,Y2,Z2]



pts = 30														# npoints for sphere 3 = 900
u = np.linspace(0, 2 * np.pi, pts)
noise = np.random.normal(0,0.25,u.shape)
u = u + noise
v = np.linspace(0, np.pi, pts)
noise = np.random.normal(0,0.25,v.shape)
u = v + noise
n_sp3 = pts*pts
X3 = centre[0] + radius3 * np.outer(np.cos(u), np.sin(v))
Y3 = centre[1] + radius3 * np.outer(np.sin(u), np.sin(v))
Z3 = centre[2] + radius3 * np.outer(np.ones(np.size(u)), np.cos(v))
X3 = X3.reshape((n_sp3,1))
Y3 = Y3.reshape((n_sp3,1))
Z3 = Z3.reshape((n_sp3,1))
ax.scatter3D(X3,Y3,Z3,c='#FFFD00');								# Yellow
class3 = np.c_[X3,Y3,Z3]

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


X_train = np.concatenate((class1,class2,class3),axis=0)
Y_train = np.ones(shape=(n_sp1 + n_sp2 + n_sp3, 1), dtype=np.int8)
for i in range(n_sp1+n_sp2+n_sp3):
	if(i<n_sp1):
		Y_train[i] = 1
	elif(i<n_sp1+n_sp2):
		Y_train[i] = 2
	else:
		Y_train[i] = 3


reds = Y_train.ravel() == 1				# helps in finding those X_train which have Y_train == 1
blues = Y_train.ravel() == 2			# helps finding those X_train which have Y_train == 2
yellows = Y_train.ravel() ==3			# helps in finding those X_train which have Y_train == 3
#print(np.shape(X_train))											# total 2700 
#print(np.shape(Y_train))


# LDA
lda = LDA(n_components = 2)
X_lda = lda.fit_transform(X_train, Y_train.ravel())
plt.scatter(X_lda[reds,0], X_lda[reds,1], c='#FF0000')
plt.scatter(X_lda[blues,0], X_lda[blues,1], c='#8500FF')
plt.scatter(X_lda[yellows,0], X_lda[yellows,1], c='#FFFD00')
plt.title("Fisher's LDA")
plt.show()


# Linear PCA
lpca = KernelPCA(n_components = 2, kernel='linear')
X_lpca = lpca.fit_transform(X_train, Y_train.ravel())
plt.scatter(X_lpca[reds,0], X_lda[reds,1], c='#FF0000')					# put color red if labelled 1
plt.scatter(X_lpca[blues,0], X_lda[blues,1], c='#8500FF')
plt.scatter(X_lpca[yellows,0], X_lda[yellows,1], c='#FFFD00')
plt.title("Linear PCA")
plt.show()


# Polynomial PCA
ppca = KernelPCA(n_components = 2, kernel='poly', degree = 5)
X_ppca = lpca.fit_transform(X_train, Y_train.ravel())
plt.scatter(X_ppca[reds,0], X_lda[reds,1], c='#FF0000')
plt.scatter(X_ppca[blues,0], X_lda[blues,1], c='#8500FF')
plt.scatter(X_ppca[yellows,0], X_lda[yellows,1], c='#FFFD00')
plt.title("Polynomial PCA")
plt.show()


# Gaussian PCA (radial basis kernel function)
gpca = KernelPCA(n_components = 2, kernel='rbf')
X_gpca = gpca.fit_transform(X_train, Y_train.ravel())
plt.scatter(X_gpca[reds,0], X_lda[reds,1], c='#FF0000')
plt.scatter(X_gpca[blues,0], X_lda[blues,1], c='#8500FF')
plt.scatter(X_gpca[yellows,0], X_lda[yellows,1], c='#FFFD00')
plt.title("Gaussian PCA")
plt.show()
