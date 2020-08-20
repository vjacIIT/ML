file1 = open("iris.data",'r')
file2 = open("iris-svm-input.txt",'w')
data = file1.readlines()

classNo = {"Iris-setosa":'1', "Iris-versicolor":'2', "Iris-virginica":'3'}

for line in data:
	words = line.split(',')						# splitting based on ','
	if(words[0]!='\n'):
		class_name=(words[-1][0:-1])			# -1 indexing gives the last element
		file2.write(classNo[class_name])
		for index in range(len(words)-1):		# last element already printed (class number)
			if(words[index]!=0.0):
				file2.write(' '+str(index+1)+':'+words[index])
		file2.write('\n')
