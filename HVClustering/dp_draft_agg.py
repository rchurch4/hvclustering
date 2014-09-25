# AD-DAG CLASS
# ad-dag.py
#
# contains:
#	- algorithm and method for clustering
#	- algorithm to create better final clusters
#		through ranking 1-D clusters
#	- clusters through draft-like process (round robin) & aggregation
#	- MAX AVG DISTANCE
#	- original greedy algorithm for comparison's sake
#
# 1-D High Variance Clustering
# Greedy Clustering Algorithm
# with cluster ranking
#
# Rob Churchill

#import
import random as rand
import numpy as np
import math
import copy as cp

################################################################################
#   EDIT THESE VALUES TO CHANGE THE SIZE OF THE DATASET AND SIZE OF CLUSTERS   #
################################################################################
n = 1000
p = 100
l = n/p
normMean = 50
normStdev = 20
resultsFile = open('results.csv', 'a')
################################################################################
# ^ EDIT THESE VALUES TO CHANGE THE SIZE OF THE DATASET AND SIZE OF CLUSTERS ^ #
################################################################################

#def algorithm(dataset, p) compute best cluster of k students
def DP (d1, size, x): 
	F = np.zeros((p+1, size))
	P = np.zeros((p+1, size), dtype=('i4,i4'))

	for j in range (1, size):
		for k in range (1, min(p, j)+1):
			t1 = F[k, j-1] + k*(p-k)*(d1[j][x] - d1[j-1][x])
			t2 = F[k-1, j-1] + (k-1)*(p-k+1)*(d1[j][x] - d1[j-1][x])

			if t1 > t2:
				F[k,j] = t1
				tup = (k, j-1)
				P[k,j] = tup
			else:
				F[k,j] = t2
				tup = (k-1, j-1)
				P[k,j] = tup

	cluster = list()

	# print P
	# print F

	k = p
	j = size-1
	next = (k,j)

	while len(cluster) < p:
		if next[0] == 0:
			cluster.append(d1[0])
			d1.remove(d1[0])

		possible = P[next[0], next[1]]
		if possible[0] == k-1:
			if possible[1] == j-1:
				cluster.append(d1[j])
				d1.remove(d1[j])
		next = possible
		k = possible[0]
		j = possible[1]

	return (cluster, F[p,size-1])

def cluster1D (d1, size, x):
	d1 = cp.deepcopy(dataset)
	clusterInst = list()
	clusterAvg = list()
	#create n/k clusters of k points each
	count = 0
	while size > p:
		count+=1
		c1 = DP(d1, len(d1), x)
		# print "Cluster " + str(count) + ": " + str(len(c1[0]))
		# print "Score: " + str(computeScore(c1[0], x)) + ", OPT = "+str(c1[1])
		# print
		clusterInst.append(c1[0])

		size = len(d1)
	
	if size > 0:
		# print "Cluster " + str(count+1) + ": " + str(len(d1))
		# print "Score: " + str(computeScore(d1, x))
		# print
		clusterInst.append(d1)

	return clusterInst

def computeTotalScore(cluster):
	score = 0
	index = 1
	while index < len(cluster[0]):
		score += computeScore(cluster, index)
		index += 1

	score = score/(len(cluster[0])-1)
	return score

	#computes the final score of a cluster
def computeScore (cluster, x):
	score = 0
	index = 0
	while index < len(cluster):
		i = cluster[index]
		currentIdx = index+1
		while currentIdx < len(cluster):
			score += math.fabs(i[x] - cluster[currentIdx][x])
			currentIdx += 1
		index +=1

	return score/len(cluster)

def computeAvgDistance (cluster):
	score = 0
	count = 0
	index = 0
	while index < len(cluster)-1:
		p1 = cluster[index]
		#print p1
		dimension = 1
		while dimension < len(p1):
			nxtIdx = index + 1
			while nxtIdx < len(cluster):
				count += 1
				score = float((((score*(count-1)) + math.fabs(p1[dimension]-cluster[nxtIdx][dimension]))/ float(count)))
				nxtIdx += 1
			dimension += 1
		index += 1

	return score

#def algorithm(dataset, k) compute best cluster of k students
def clustering (d1, size, p, x): 
	#create empty cluster
	cluster = list()
	#add smartest person in dataset to start
	cluster.append(d1[len(d1)-1])
	d1.remove(d1[len(d1)-1])
	while len(cluster) < p:
		#index and score of best point so far
		maxDist = 0
		maxIdx = 0
		#index in d1
		index = 0
		while index < len(d1):
			newDist = 0
			clusterIdx = 0
			#get distance from point d1[index] to all points in cluster
			while clusterIdx < len(cluster):
				newDist += math.fabs(cluster[clusterIdx][x] - d1[index][x])
				clusterIdx+=1

			#if its distance is higher than all others, change maxDist and maxIdx to reflect that
			if ( newDist > maxDist ):
				maxDist = newDist
				maxIdx = index
			index += 1
		#add the point with highest avg distance to the cluster, remove from d1
		cluster.append(d1[maxIdx])
		d1.remove(d1[maxIdx])

	return cluster

def cluster1DGreedy (d1, size, p, x):
	d1 = cp.deepcopy(dataset)
	clusterInst = list()
	clusterAvg = list()
	#create n/k clusters of k points each
	count = 0
	while size > p-1:
		count+=1
		c1 = clustering(d1, len(d1), p, x)
		#print "Cluster " + str(count) + ": " + str(c1)
		#print "Score: " + str(computeScore(c1, x))
		#print
		clusterInst.append(c1)
		size = len(d1)
	return clusterInst

#create dataset
dataset = list()
nextId = 0
size = 0

print "N = "+str(n)+", P = "+str(p)

#create dataset according to a normal distribution with mean 50 and stdev 20
while size < n:
	size += 1
	meth = int(rand.normalvariate(normMean, normStdev)) #random number, 0-100
	english = int(rand.normalvariate(normMean, normStdev)) #random number, 0-100
	history = int(rand.normalvariate(normMean, normStdev)) #random number, 0-100
	nextStudent = (nextId, meth, english, history)
	dataset.append(nextStudent)
	nextId += 1

#sort the dataset by intelligence
randomCopy = cp.deepcopy(dataset)
dataset = sorted(dataset, key=lambda student: student[1])

originalCopy = cp.deepcopy(dataset)
l1 = cluster1D(dataset, size, 1)
l4 = cluster1DGreedy(dataset, size, p, 1)

dataset = sorted(dataset, key=lambda student: student[2])
l2 = cluster1D(dataset, size, 2)
l5 = cluster1DGreedy(dataset, size, p, 2)

dataset = sorted(dataset, key=lambda student: student[3])
l3 = cluster1D(dataset, size, 3)
l6 = cluster1DGreedy(dataset, size, p, 3)
clusters = [l1, l2, l3]
greedys = [l4, l5, l6]

#empty matrix [n, n]
F = np.zeros((n, n))

#for each 1-d clustering
for y in clusters:
	idx = 0
	#take each cluster, with factor higher for earlier clusters
	while idx < len(y):
		c1 = y[idx]
		factor = len(y) - math.pow(6, idx)
		if factor < 0:
			factor = 0
		#and for each pair in the cluster
		i1 = 0
		while i1 < len(c1)-1:
			i2 = i1+1
			while i2 < len(c1):
				#add the factor of the cluster to their partnership F[x, q]
				F[c1[i1], c1[i2]] += factor/(len(clusters)*p)
				F[c1[i2], c1[i1]] += factor/(len(clusters)*p)
				i2+=1
			i1+=1
		idx+=1

dataset = sorted(dataset, key=lambda student: student[0])

gryffindor = cp.deepcopy(F)

clusters = list()
while len(clusters) < l:
	clusters.append(list())

while len(clusters[l-1]) < p:
	#print "next round"
	for i in clusters:
		firstpoints = np.unravel_index(np.argmax(F), (n,n))
		i.append(dataset[firstpoints[0]])
		i.append(dataset[firstpoints[1]])
		#print "current cluster "+str(i)
		#print "next pick"
		maxIdx = 0
		maxVal = 0
		index = 0
		while index < min(len(i), 2):
			test = i[index][0]
			possIdx = np.unravel_index(np.argmax(F[:, test]), (n,n))
			if possIdx[1] == test:
				F[possIdx[1], test] = -1
				possIdx = np.unravel_index(np.argmax(F[:, test]), (n,n))
			possMax = F[possIdx[1], test]
			if possMax >= maxVal:
				maxIdx = (possIdx[1], test)
				maxVal = possMax
			index += 1
		i.append(dataset[maxIdx[0]])
		#print "picked "+ str(dataset[maxIdx])
		F[maxIdx[0], :] = -1

# for i in clusters:
# 	print i
print "DP Draft Scores: "

scores = list()
avgscore = 0
avgdistance = 0
index = 0
for i in clusters:
	x = computeTotalScore(i)
	zl = computeAvgDistance(i)
	avgdistance += zl
	avgscore += x
	scores.append(x)
	#print "Cluster "+ str(index)+" Score: "+str(x)+", Size: "+str(len(i))
	index += 1

avgscore = avgscore/l
avgdistance = avgdistance/l
var = 0
for i in scores:
	var += math.pow(avgscore - i, 2)
stdev = math.sqrt(var)
print "Average Score: "+str(avgscore)
resultsFile.write(str(avgscore)+",")
print "Standard Deviation: "+str(stdev)
resultsFile.write(str(stdev)+",")
print "Average Distance: "+str(avgdistance)
resultsFile.write(str(avgdistance)+",")
print

##############################################
# 	 DP Aggregate Clustering algorithm   	 #
# 											 #
# 		    For Comparison's sake			 #
##############################################

dataset = sorted(dataset, key=lambda student: student[0])

F = gryffindor

clusters = list()
while len(clusters) < l:
	nextC = list()
	while len(nextC) < p:
		firstpoints = np.unravel_index(np.argmax(F), (n,n))
		nextC.append(dataset[firstpoints[0]])
		nextC.append(dataset[firstpoints[1]])
		#print "current cluster "+str(i)
		#print "next pick"
		maxIdx = 0
		maxVal = 0
		index = 0
		while index < min(len(nextC), 2):
			test = nextC[index][0]
			possIdx = np.unravel_index(np.argmax(F[:, test]), (n,n))
			if possIdx[1] == test:
				F[possIdx[1], test] = -1
				possIdx = np.unravel_index(np.argmax(F[:, test]), (n,n))
			possMax = F[possIdx[1], test]
			if possMax >= maxVal:
				maxIdx = (possIdx[1], test)
				maxVal = possMax
			index += 1
		nextC.append(dataset[maxIdx[0]])
		#print "picked "+ str(dataset[maxIdx])
		F[maxIdx[0], :] = -1
	clusters.append(nextC)

# for i in clusters:
# 	print i
print "DP Aggregate Scores: "

scores = list()
avgscore = 0
avgdistance = 0
index = 0
for i in clusters:
	x = computeTotalScore(i)
	zl = computeAvgDistance(i)
	avgdistance += zl
	avgscore += x
	scores.append(x)
	#print "Cluster "+ str(index)+" Score: "+str(x)+", Size: "+str(len(i))
	index += 1

avgscore = avgscore/l
avgdistance = avgdistance/l
var = 0
for i in scores:
	var += math.pow(avgscore - i, 2)
stdev = math.sqrt(var)
print "Average Score: "+str(avgscore)
resultsFile.write(str(avgscore)+",")
print "Standard Deviation: "+str(stdev)
resultsFile.write(str(stdev)+",")
print "Average Distance: "+str(avgdistance)
resultsFile.write(str(avgdistance)+",")
print

##############################################
#  Original Greedy M-D Clustering Algorithm  #
# 											 #
# 		    For Comparison's sake			 #
##############################################

#def algorithm(dataset, k) compute best cluster of k students
def clusterOriginal (originalCopy, size, p): 
	#create empty cluster
	cluster = list()
	#add smartest person in dataset to start
	cluster.append(originalCopy[len(originalCopy)-1])
	originalCopy.remove(originalCopy[len(originalCopy)-1])
	while len(cluster) < p:
		#index and score of best point so far
		maxDist = 0
		maxIdx = 0
		#index in dataset
		index = 0
		while index < len(originalCopy):
			newDist = 0
			clusterIdx = 0
			#get distance from point dataset[index] to all points in cluster
			while clusterIdx < len(cluster):
				i = 1
				while i < len(originalCopy[0]):
					newDist += math.fabs(cluster[clusterIdx][i] - originalCopy[index][i])
					i+=1
				clusterIdx+=1

			#if its distance is higher than all others, change maxDist and maxIdx to reflect that
			if ( newDist > maxDist ):
				maxDist = newDist
				maxIdx = index
			index += 1
		#add the point with highest avg distance to the cluster, remove from dataset
		cluster.append(originalCopy[maxIdx])
		originalCopy.remove(originalCopy[maxIdx])

	return (cluster, 0)

print "Greedy M-D Scores: "

scores = list()
avgscore = 0
avgdistance = 0
count = 0
while count < l:
	count+=1
	c1 = clusterOriginal(originalCopy, len(originalCopy), p)
	x = computeTotalScore(c1[0])
	zl = computeAvgDistance(c1[0])
	avgscore += x
	avgdistance += zl
	scores.append(x)
	#print "Cluster " + str(count) + " Score: " + str(x)+", Size: "+str(len(c1))

avgscore = avgscore/l
avgdistance = avgdistance/l
var = 0
for i in scores:
	var += math.pow(avgscore - i, 2)
stdev = math.sqrt(var)
print "Average Score: "+str(avgscore)
resultsFile.write(str(avgscore)+",")
print "Standard Deviation: "+str(stdev)
resultsFile.write(str(stdev)+",")
print "Average Distance: "+str(avgdistance)
resultsFile.write(str(avgdistance)+",")
print

########################################
#			Random Clustering 		   #
#			for Comparison 			   #
#									   #
########################################

randomClustering = list()
while len(randomClustering) < l:
	cluster1 = list()
	while len(cluster1) < p:
		rndnum = rand.randint(0, len(randomCopy)-1)
		next = randomCopy[rndnum]
		randomCopy.remove(randomCopy[rndnum])
		cluster1.append(next)
	#print cluster1
	randomClustering.append(cluster1)

print "Random Scores: "

scores = list()
avgscore = 0
avgdistance = 0
for c1 in randomClustering:
	x = computeTotalScore(c1)
	zl = computeAvgDistance(c1)
	avgscore += x
	avgdistance += zl
	scores.append(x)
	#print "Cluster " + str(count) + " Score: " + str(x)+", Size: "+str(len(c1))

avgscore = avgscore/l
avgdistance = avgdistance/l
var = 0
for i in scores:
	var += math.pow(avgscore - i, 2)
stdev = math.sqrt(var)
print "Average Score: "+str(avgscore)
resultsFile.write(str(avgscore)+",")
print "Standard Deviation: "+str(stdev)
resultsFile.write(str(stdev)+",")
print "Average Distance: "+str(avgdistance)
resultsFile.write(str(avgdistance)+'\n')
print



