import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

comparison = []
N = 2										# Hidden states
M = 27										# Visible states

mmap = {}									# mapping each character to number for states
for i in range(0,M-1):
	mmap[chr(i+65)] = i;

mmap[' ']=M-1
#print(mmap)


##			|````
##			|----
##			|____ VALUATE
# Given A, B, pie, T and X, evaluate P(vT | theta)
def Evaluate(A,B,alpha,c,pie,T,X):
	for i in range(0,N):
		alpha[0][i] = pie[0][i]*B[i][mmap[X[0]]]
		c[0][0] = c[0][0] + alpha[0][i]
		
	c[0][0] = 1/c[0][0]
	for i in range(0,N):
		alpha[0][i] = c[0][0]*alpha[0][i]

	for t in range(1,T):
		for i in range(0,N):
			for j in range(0,N):
				alpha[t][i] = alpha[t][i] + alpha[t-1][j]*A[j][i]
			
			alpha[t][i] = alpha[t][i]*B[i][mmap[X[t]]]
			c[0][t] = c[0][t] + alpha[t][i]
		
		## scale alpha[t][i], very important otherwise alpha values will become 0
		c[0][t]=1/c[0][t]
		for i in range(0,N):
			alpha[t][i] = c[0][t]*alpha[t][i]
		
	log_prob_O_lamb = 0.0
	for i in range(0,T):
		log_prob_O_lamb = log_prob_O_lamb - np.log(c[0][i])
	comparison.append(log_prob_O_lamb)
	return log_prob_O_lamb



##			|
##			|
##			|____EARN
# Given dimenstion M, N, T, learn the transitions
def Learn(A,B,pie,T,X):
# 1.  Initialization
	alpha = np.zeros(shape=(T,N),dtype=np.float64)
	beta = np.zeros(shape=(T,N),dtype=np.float64)
	Dgamma = np.zeros(shape=(T,N,N),dtype=np.float64)
	gamma = np.zeros(shape=(T,N),dtype=np.float64)
	c = np.zeros(shape=(1,T),dtype=np.float64)

	maxIters = 10
	#iters = 0
	oldLogProb = float('-Inf')

	for iters in range(0,maxIters):
# 2.  alpha pass
		
		c[0][0] = 0.0
		## compute alpha[0][i]
		for i in range(0,N):
			alpha[0][i] = pie[0][i]*B[i][mmap[X[0]]]
			c[0][0] = c[0][0] + alpha[0][i]

		## scale alpha[0][i]	
		if(c[0][0]!=0.0):
			c[0][0] = 1/c[0][0]
		for i in range(0,N):
			alpha[0][i] = c[0][0]*alpha[0][i]

		## compute alpha[t][i]
		for t in range(1,T):
			c[0][t] = 0.0
			
			for i in range(0,N):
				alpha[t][i] = 0.0
				
				for j in range(0,N):
					alpha[t][i] = alpha[t][i] + alpha[t-1][j]*A[j][i]
				
				alpha[t][i] = alpha[t][i]*B[i][mmap[X[t]]]
				c[0][t] = c[0][t] + alpha[t][i]
			
			## scale alpha[t][i], very important otherwise alpha values will become 0
			if(c[0][t]!=0.0):
				c[0][t]=1/c[0][t]
			for i in range(0,N):
				alpha[t][i] = c[0][t]*alpha[t][i]
				
				
# 3. beta pass
		
		## Let beta[T-1][i] = 1, scaled by c[T-1]
		for i in range(0,N):
			beta[T-1][i] = c[0][T-1]
			
		for t in range(T-2,-1,-1):
			for i in range(0,N):
				beta[t][i] = 0.0
				for j in range(0,N):
					beta[t][i] = beta[t][i] + A[i][j]*B[j][mmap[X[t+1]]]*beta[t+1][j]
				## scale beta[t][i] with same scale factor as alpha[t][i]
				beta[t][i] = c[0][t]*beta[t][i]
				

# 4. Compute Dgamma[t][i][j] and gamma[t][i]
		
		for t in range(0,T-1):
			for i in range(0,N):
				gamma[t][i] = 0.0
				for j in range(0,N):
					Dgamma[t][i][j] = alpha[t][i]*A[i][j]*B[j][mmap[X[t+1]]]*beta[t+1][j]
					gamma[t][i] = gamma[t][i] + Dgamma[t][i][j]
					
		# Special case for gamma[T-1][i]
		for i in range(0,N):
			gamma[T-1][i] = alpha[T-1][i]
			

# 5. Re-estimate A, B and pie

		# re-estimate pie
		for i in range(0,N):
			pie[0][i] = gamma[0][i]

		# re-estimate A
		for i in range(0,N):
			denom = 0.0
			for t in range(0,T-1):
				denom = denom+gamma[t][i]
			
			for j in range(0,N):
				numer = 0.0
				for t in range(0,T-1):
					numer = numer + Dgamma[t][i][j]
				
				if(denom!=0):
					A[i][j] = numer/denom

				
		# re-estimate B
		for i in range(0,N):
			denom = 0.0
			for t in range(0,T):
				denom = denom+gamma[t][i]
			
			for j in range(0,M):
				numer = 0
				for t in range(0,T):
					if(mmap[X_tr[t]]==j):
						numer = numer + gamma[t][i]
				
				if(denom!=0):
					B[i][j] = numer/denom
				
# 6. Compute log[P(O|lambda)]
		logProb = 0.0
		for i in range(0,T):
			logProb = logProb + np.log(c[0][i])
		logProb = -logProb

# 7. To iterate or not
		#iters = iters + 1
		if(logProb>oldLogProb):
			oldLogProb = logProb
		else:
			break

	return A,B,pie,alpha,beta,c



file = open('hmm-train.txt','r')
data = file.readlines()

X_tr = ""
for line in data:
	for letter in line:
		if(letter.isalpha())==True:
			X_tr=X_tr+letter.upper()
		else:
			X_tr=X_tr+' '
		
X_tr = ' '.join(X_tr.split())				# removing multiples spaces		
T = len(X_tr)								# Training input size
#print(X_tr)



# Learning
A = (np.ones(shape=(N,N),dtype=np.float64))/N
e = pow(10,-3)
for i in range(0,N):
	msum = 0.0
	for j in range(0,N-1):
		k = np.random.choice([-e,e],p=[0.5,0.5])
		msum = msum + k
		A[i][j]=A[i][j] + k
	A[i][N-1]=A[i][N-1]-msum
#print(A)

e = pow(10,-5)
B = (np.ones(shape=(N,M),dtype=np.float64))/M
for i in range(0,N):
	msum = 0.0
	for j in range(0,M-1):
		k = np.random.choice([-e,e],p=[0.5,0.5])
		msum = msum + k
		B[i][j] = B[i][j] + k
	B[i][M-1] = B[i][M-1]-msum
#print(B)

pie = (np.ones(shape=(1,N),dtype=np.float64))/N
msum = 0.0
for i in range(0,N-1):
	k = np.random.choice([-e,e],p=[0.5,0.5])
	msum = msum + k
	pie[0][i] = pie[0][i] + k
pie[0][N-1] = pie[0][N-1]-msum
#print(pie)


# Learning the HMM
print("Learning without natural hmm initialization")
A,B,pie,alpha,beta,c = Learn(A,B,pie,T,X_tr)	
print("starting state probability\n",pie)
print("\nhidden states probability\n",A)
print("\nhidden to visible state probabilties\n",B)


# Evaluation on training set
log_prob_O_lamb_train = 0.0
for i in range(0,T):
	log_prob_O_lamb_train = log_prob_O_lamb_train - np.log(c[0][i])
print("\nLog Probability of training set",log_prob_O_lamb_train)
comparison.append(log_prob_O_lamb_train)
file.close()




# Evaluate on natural hmm
B_nat = np.ones(shape=(N,M),dtype=np.float64)/M
A_nat = np.zeros(shape=(N,N),dtype=np.float64)
A_nat[0][0] = 0.7
A_nat[1][1] = 0.3
A_nat[0][1] = 0.7
A_nat[1][0] = 0.3
alpha_nat = np.zeros(shape=(T,N),dtype=np.float64)
c_nat = np.zeros(shape=(1,T),dtype=np.float64)
pie_nat = (np.ones(shape=(1,N),dtype=np.float64))/N
print("\nLog Probability of natural hmm without learning",Evaluate(A_nat,B_nat,alpha_nat,c_nat,pie_nat,T,X_tr))





# Initializing the A, B, pie with natural HMM and learning again
A_nat,B_nat,pie_nat,alpha_nat,beta_nat,c_nat = Learn(A_nat,B_nat,pie_nat,T,X_tr)
# Evaluation on training set with initialization with natural hmm
log_prob_O_lamb_train = 0.0
for i in range(0,T):
	log_prob_O_lamb_train = log_prob_O_lamb_train - np.log(c_nat[0][i])
print("\nLog Probability of training set after learning with natural hmm initialization",log_prob_O_lamb_train)
comparison.append(log_prob_O_lamb_train)
file.close()




# Evaluation on test set
file = open('hmm-test.txt','r')
data = file.readlines()

X_tst = ""
for line in data:
	for letter in line:
		if(letter.isalpha())==True:
			X_tst=X_tr+letter.upper()
		else:
			X_tst=X_tst+' '
		
X_tst = ' '.join(X_tst.split())				# removing multiples spaces		
T2 = len(X_tst)								# Test input size

alpha2 = np.zeros(shape=(T2,N),dtype=np.float64)
c2 = np.zeros(shape=(1,T2),dtype=np.float64)
print("\nLog Probability of test hmm",Evaluate(A,B,alpha2,c2,pie,T2,X_tst))
file.close()

names = ['train set','natural hmm','train with natural hmm','test set']
plt.xlabel("Type")
plt.ylabel("Evaluation value")
plt.bar(names,comparison)
plt.show()
