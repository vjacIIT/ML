import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = open('train.csv','r')
data = file.readlines()

X_tr = []
Y_tr = []
lineNo=1
for line in data:
	words = line.split(',')
	if(lineNo==1):
		lineNo=2
		continue
	X_tr.append(float(words[0]))
	Y_tr.append(float(words[1][:-1]))


dpoints = len(X_tr)
validation_size = int(dpoints/4)
train_size = dpoints-validation_size

X_train = np.zeros(shape=(train_size,11), dtype=np.float64)
Y_train = np.zeros(shape=(train_size,1), dtype=np.float64)
X_validation = np.zeros(shape=(validation_size,11), dtype=np.float64)
Y_validation = np.zeros(shape=(validation_size,1), dtype=np.float64)
X_full = np.zeros(shape=(dpoints,11), dtype=np.float64)
Y_full = np.zeros(shape=(dpoints,1), dtype=np.float64)


for i in range(dpoints):
	for p in range(11):
		X_full[i][p] = pow(X_tr[i],p)
	Y_full[i] = Y_tr[i]

for i in range(train_size):							# 3/4 for training, 1/4 for validation
	for p in range(11):
		X_train[i][p] = pow(X_tr[i],p)
	Y_train[i] = Y_tr[i]

for i in range(validation_size):
	for p in range(11):
		X_validation[i][p] = pow(X_tr[i+train_size],p)
	Y_validation[i] = Y_tr[i+train_size]
	
Lambda= 100.0
validation_loss = 100000
t_loss = []
v_loss = []
diff_l = []

for i in range(100000):
	lbda = i/10000.0
	diff_l.append(lbda)
	Identity = np.ones(shape=(11,11), dtype=np.float64)
	inv = np.linalg.inv((lbda*Identity) + np.dot(X_train.T,X_train))		# for inverse
	w = np.dot(inv,np.dot(X_train.T,Y_train))
	
	#print(lbda)
	#print(w)

	train_loss = np.linalg.norm(np.dot(X_train,w) - Y_train) + lbda*np.linalg.norm(w)
	t_loss.append(train_loss)										# appending the trinaing loss
	#print(train_loss)
	
	least_sq_loss = np.linalg.norm(np.dot(X_validation,w) - Y_validation)
	reg_loss = lbda*np.linalg.norm(w)
	loss = least_sq_loss + reg_loss
	v_loss.append(loss)
	#print(loss)
	 
	if(validation_loss>loss):
		validation_loss = loss
		Lambda = lbda
	#print(loss)
	
	
plt.plot(diff_l,t_loss)
plt.ylabel('training_loss')
plt.xlabel('lambda value')
plt.show()

plt.plot(diff_l,v_loss)
plt.ylabel('validation_loss')
plt.xlabel('lambda value')
plt.show()

print(Lambda, validation_loss)
print('\n')

# Tried different lambda on same validation set (1/4 of training set)
# Best lambda is coming out to be 2.8676, validation_loss = 144.051

# As we increase lambda our training loss gets higher (2.77 -> 10.86), but our validation_loss decreases sharply because we are making our model more rigrous to new data (394.424 -> 144.051)


# Now loss on full training dataset 
w_param = np.zeros(shape=(11,1), dtype=np.float64)
Identity = np.ones(shape=(11,11), dtype=np.float64)
inv = np.linalg.inv((Lambda*Identity) + np.dot(X_full.T,X_full))		# for inverse
w_param = np.dot(inv,np.dot(X_full.T,Y_full))
total_loss = np.linalg.norm(np.dot(X_full,w_param) - Y_full) + lbda*np.linalg.norm(w_param)
print(w_param,total_loss)

# total_loss is coming out to be 8.182 on full training file
file.close()


'''
w_param = [[-4.08305847e-01],
		   [ 2.55486074e-01],
		   [ 2.01056899e-02],
		   [ 5.04332245e-02],
		   [ 1.26835695e-02],
		   [-7.85079707e-03],
		   [-2.49761982e-03],
		   [ 4.08142658e-04],
		   [ 1.45730168e-04],
		   [-4.97537552e-06],
		   [-2.75538768e-06]]
'''

file = open('testX.csv','r')
data = file.readlines()

X_tt = []
Y_tt = []
lineNo=1
for line in data:
	words = line.split()
	if(lineNo==1):
		lineNo=2
		continue
	X_tt.append(float(words[0]))
	
test_size = len(X_tt)
X_test = np.zeros(shape=(test_size,11), dtype=np.float64)
Y_test = np.zeros(shape=(test_size,1), dtype=np.float64)

for i in range(test_size):							# 3/4 for training, 1/4 for validation
	for p in range(11):
		X_test[i][p] = pow(X_tt[i],p)
		
Y_test = np.dot(X_test,w_param)
# print(Y_test)
file.close()

for i in range(len(X_tt)):
	Y_tt.append(round(Y_test[i][0],6))


dict = {'Xts':X_tt, 'Yp':Y_tt}
df = pd.DataFrame(dict)
df.to_csv('submission.csv', index=False)
