import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.utils import shuffle
import matplotlib.image as mpimg

color_theme = np.array(['#FF0000','#8500FF','#FFFD00'])			# Red, Blue and Yellow

def printCluster(X,Y,title):
	plt.title(title)
	plt.scatter(X[:,0],X[:,1],c=color_theme[Y])					# Color is based on color_theme
	return


def printOriginal(X1,X2,X3,flag):
	X = np.concatenate((X1,X2,X3))
	if(flag==0):												# print colored
		plt.title('Original Data')
		plt.plot(X1[:,0],X1[:,1],'ro')
		plt.plot(X2[:,0],X2[:,1],'bo')
		plt.plot(X3[:,0],X3[:,1],'yo')
	else:
		plt.title('Original Data without classification')
		plt.plot(X[:,0],X[:,1],'ro')


def call_Kmeans(X,inits,n_inits,cnt,title,max_iters):
	print("Type of initialization\n",inits)
	kmeans = KMeans(n_clusters=3,init=inits,n_init=n_inits,max_iter=max_iters)
	Y = kmeans.fit_transform(X)
	print("Score on",title,kmeans.score(X))
	print()
	plt.subplot(2,2,cnt)
	printCluster(X,kmeans.labels_,title)
	return X, kmeans.labels_


def call_GMM(X,n_inits,n_iters):
	gmm = GaussianMixture(n_components=3,max_iter=n_iters,n_init=n_inits)
	Y = gmm.fit_predict(X)
	print("Log-likelihood score on GMM",gmm.score(X))
	print()
	plt.subplot(1,2,2)
	printCluster(X,Y,'GMM')	
	return X,Y


def compressImage(image,w,h,d,n_colors):
	
	# Because (w*h,n) is very big ((2304000,3) in my case), not possible to apply k-means to whole image
	# Hence we have taken shuffled part of image to fit the training
	image_sample = shuffle(image)[:1000]
	kmeans = KMeans(n_clusters=n_colors,init='k-means++',n_init=5,max_iter=100).fit(image_sample)

	# These are the labels accquired from learned k-means cluster means
	labels = kmeans.predict(image)
	#print(np.shape(labels))

	# Learned clusters (30,3)
	centres = kmeans.cluster_centers_
	#print(np.shape(centres))

	# Reverting back to 3D image
	new_image = np.zeros((w,h,d))
	index = 0											# for each label
	for i in range(w):
		for j in range(h):
			new_image[i][j] = centres[labels[index]]
			index = index+1

	return new_image


n_clusters = 3
n_features = 2
X_1 = np.random.multivariate_normal(mean=[4, 0], cov=[[1, 0], [0, 1]], size=75)
X_2 = np.random.multivariate_normal(mean=[6, 6], cov=[[2, 0], [0, 2]], size=250)
X_3 = np.random.multivariate_normal(mean=[1, 5], cov=[[1, 0], [0, 2]], size=20)
plt.subplot(2,2,1)
printOriginal(X_1,X_2,X_3,0)

X1 = np.concatenate((X_1,X_2,X_3))
X1,labels1 = call_Kmeans(X1,'k-means++',100,2,'Best_Kmeans',300)

X2 = np.concatenate((X_1,X_2,X_3))
high = 8
low = 2
randArray = (high-low)*np.random.rand(n_clusters,n_features) + low
X2,labels2 = call_Kmeans(X2,randArray,1,3,'Random_Initializatin_1',10)

X3 = np.concatenate((X_1,X_2,X_3))
randArray = np.random.rand(n_clusters,n_features)
X3,labels3 = call_Kmeans(X3,randArray,1,4,'Random_Initialization_2',10)
plt.show()



### GMM
X4 = np.concatenate((X_1,X_2,X_3))
plt.subplot(1,2,1)
printOriginal(X_1,X_2,X_3,0)
X4,Y = call_GMM(X4,100,300)
plt.show()


# Comparison between GMM and K-means
plt.subplot(2,2,1)								# Data without colors
printOriginal(X_1,X_2,X_3,1)
plt.subplot(2,2,2)								# Data with colors
printOriginal(X_1,X_2,X_3,0)
plt.subplot(2,2,3)								# Kmeans
printCluster(X1,labels1,'K-Means')
plt.subplot(2,2,4)								# GMM
printCluster(X4,Y,'GMM')
plt.show()




# Image compression
img = mpimg.imread('Vaibhav_Jindal.jpeg')
img = img/255
#print(np.shape(img))
#print(img)
w,h,d = np.shape(img)
image_array = np.reshape(img,(w*h,d))				# converting image to 2D
plt.figure(1)
plt.title('Original image')
plt.imshow(img)

n_colors = 10
image1 = compressImage(image_array,w,h,d,n_colors)
plt.figure(2)
plt.title('Compressed image, n_colors = 10')
plt.imshow(image1)

n_colors = 30
image2 = compressImage(image_array,w,h,d,n_colors)
plt.figure(3)
plt.title('Compressed image, n_colors = 30')
plt.imshow(image2)
plt.show()

print('Dimension of original image',np.shape(img))
print('Dimension of first compressed image',np.shape(image1))
print('Dimenstion of second compressed image',np.shape(image2))
