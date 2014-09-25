# GREEDYRANK2 CLASS
# greedyrank2.py
#
# contains:
#	- algorithm and method for clustering
#	- algorithm to create better final clusters
#		through ranking 1-D clusters
#	- CHANGED DISTANCE FUNCTION FROM AVG DISTANCE TO MAX LEARNING
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

#def algorithm(dataset, k) compute best cluster of k students
def clustering (d1, size, k, x, divFact): 
	#create empty cluster
	cluster = list()
	#add smartest person in dataset to start
	blah = int(len(d1)/2)
	mean = d1[blah][x]
	meanIdx = blah
	cluster.append(d1[len(d1)-1])
	d1.remove(d1[len(d1)-1])
	cluster.append(d1[blah])
	d1.remove(d1[blah])
	while len(cluster) < k:
		#index and score of best point so far
		maxDist = 0
		maxIdx = -1
		index = 0
		while index < min(meanIdx, len(d1)):
			intel = d1[index][x]
			if intel < mean:
				if intel > mean/divFact:
					learning = mean - intel
					if maxDist < learning:
						maxIdx = index
						maxDist = learning
			index += 1

		if maxIdx == -1:
			maxIdx = 0
		cluster.append(d1[maxIdx])
		d1.remove(d1[maxIdx])
	return cluster

def computeTotalScore(cluster, divFact):
	score = 0
	index = 1
	while index < len(cluster[0]):
		score += computeScore(cluster, index, divFact)
		index += 1

	score = score/(len(cluster[0])-1)
	return score

	#computes the final average distance of a cluster
def computeScore (cluster, x, divFact):
	score = 0
	currentIdx = 1
	while currentIdx < len(cluster[0]):
		rep = getRep(cluster, currentIdx)
		for i in cluster:
			val = i[currentIdx]
			if rep > val:
				if val > rep/divFact:
					score += rep-val
				else:
					score += rep/divFact - val
		currentIdx += 1

	return score/((len(cluster[0])-1)*len(cluster))

def getRep (cluster, x):
	rep = 0
	maxVal = 0
	maxVal2 = 0
	for i in cluster:
		if i[x] > maxVal:
			maxVal2 = maxVal
			maxVal = i[x]
		else:
			if i[x] > maxVal2:
				maxVal2 = i[x]
	if maxVal > 70:
		return maxVal2
	else: 
		print "No good teacher"
		return -100

def cluster1D (d1, size, k, x, divFact):
	d1 = cp.deepcopy(dataset)
	clusterInst = list()
	#create n/k clusters of k points each
	count = 0
	while size > k-1:
		count+=1
		c1 = clustering(d1, len(d1), k, x, divFact)
		#print "Cluster " + str(count) + ": " + str(c1)
		#print "Score: " + str(computeScore(c1, x))
		#print
		clusterInst.append(c1)
		size = len(d1)

	return clusterInst

# this is NOT CORRECT
def getStartingIdx (F):
	idx = np.argmax(F)
	return np.unravel_index(idx, (n,n))

#incomplete
def clusterRankwise (F, n, k):
	finalClusters = list()
	while (len(finalClusters) < n/k):
		nextCluster = list()
		while (len(nextCluster) < k):
			startingIdx = getStartingIdx(F)
			#add the two points to nextCluster
			#find the argmax of those two axes
			#add that point to nextCluster
			#find argmax of those 3 axes, repeat until cluster full


################################################################################
#   EDIT THESE VALUES TO CHANGE THE SIZE OF THE DATASET AND SIZE OF CLUSTERS   #
################################################################################
n = 200
k = 20
p = n/k
divFact = 2
################################################################################
# ^ EDIT THESE VALUES TO CHANGE THE SIZE OF THE DATASET AND SIZE OF CLUSTERS ^ #
################################################################################

#create dataset
dataset = list()
nextId = 0
size = 0

#create dataset according to a normal distribution with mean 50 and stdev 20
while size < n:
	size += 1
	meth = int(rand.normalvariate(50, 20)) #random number, 0-100
	english = int(rand.normalvariate(50, 20)) #random number, 0-100
	history = int(rand.normalvariate(50, 20)) #random number, 0-100
	nextStudent = (nextId, meth, english, history)
	dataset.append(nextStudent)
	nextId += 1

#sort the dataset by intelligence
randomCopy = cp.deepcopy(dataset)
dataset = sorted(dataset, key=lambda student: student[1])
originalCopy = cp.deepcopy(dataset)
l1 = cluster1D(dataset, size, k, 1, divFact)
dataset = sorted(dataset, key=lambda student: student[2])
l2 = cluster1D(dataset, size, k, 2, divFact)
dataset = sorted(dataset, key=lambda student: student[3])
l3 = cluster1D(dataset, size, k, 3, divFact)
clusters = [l1, l2, l3]

#empty matrix [n, n]
F = np.zeros((n, n))

#for each 1-d clustering
for y in clusters:
	idx = 0
	#take each cluster, with factor higher for earlier clusters
	while idx < len(y):
		#print len(y)
		c1 = y[idx]
		#print c1
		factor = len(y) - idx
		if factor < 0:
			factor = 0
		#and for each pair in the cluster
		i1 = 0
		while i1 < len(c1)-1:
			i2 = i1+1
			while i2 < len(c1):
				#add the factor of the cluster to their partnership F[x, q]
				hm1 = c1[i1][0]
				hm2 = c1[i2][0]
				F[hm1, hm2] += factor
				F[c1[i2][0], c1[i1][0]] += factor
				i2+=1
			i1+=1
		idx+=1

#for n/k clusters, take the highest pair and place them in a cluster together, s.t.
#neither point is already in a cluster

clusters = list()
while len(clusters) < n/k:
	indices = list()
	idx = getStartingIdx(F)
	indices.append(idx[0])
	indices.append(idx[1])
	F[idx] = -1
	F[(idx[1], idx[0])] = -1
	while len(indices) < k:
		maxIdxValue = -1
		maxIdx = -1
		for i in indices:
			nextIdx = np.unravel_index(np.argmax(F[i, :]), (n,n))
			nextIdxValue = F[nextIdx]
			if (nextIdxValue > maxIdxValue):
				maxIdxValue = nextIdxValue
				maxIdx = nextIdx
		if maxIdx == -1:
			maxIdx = getStartingIdx(F)
		F[maxIdx] = -1
		F[(maxIdx[1], maxIdx[0])] = -1
		if maxIdx[0] in indices:
			indices.append(maxIdx[1])
		else:
			if maxIdx[1] in indices:
				indices.append(maxIdx[0])
			else:
				indices.append(maxIdx[1])
				if len(indices) == k:
					indices.append(maxIdx[0])
	if len(indices) > k:
		indices.remove(indices[len(indices)-1])
	c = list()
	for i in indices:
		c.append(dataset[i])
	clusters.append(c)

print "Greedy Ranking Scores: "

scores = list()
avgscore = 0
index = 0
for i in clusters:
	index += 1
	x = computeTotalScore(i, divFact)
	avgscore += x
	scores.append(x)
	#print "Cluster "+ str(index)+" Score: "+str(x)+", Size: "+str(len(i))

avgscore = avgscore/(n/k)
var = 0
for i in scores:
	var += math.pow(avgscore - i, 2)
stdev = math.sqrt(var)
print "Average Score: "+str(avgscore)
print "Standard Deviation: "+str(stdev)
print

##############################################
# Original Greedy 1-D Clustering Algorithm   #
# 			Same as in greedy.py			 #
# 		    For Comparison's sake			 #
##############################################

#def algorithm(dataset, k) compute best cluster of k students
def clusterOriginal (originalCopy, size, k, divFact): 
	#create empty cluster
	cluster = list()
	#add smartest person in dataset to start
	mean = 70
	cluster.append(originalCopy[len(originalCopy)-1])
	originalCopy.remove(originalCopy[len(originalCopy)-1])
	while len(cluster) < k:
		#index and score of best point so far
		maxDist = 0
		maxIdx = -1
		index = 0
		while index < len(originalCopy):
			nextPoint = originalCopy[index]
			x = 1
			learning = 0
			while x < len(nextPoint):
				intel = nextPoint[x]
				if intel < mean:
					if intel > mean/divFact:
						nextLearn = mean - intel
						if nextLearn > 0:
							learning += nextLearn
				x+=1
			if maxDist < learning:
				maxIdx = index
				maxDist = learning
			index += 1

		if maxIdx == -1:
			maxIdx = 0
		cluster.append(originalCopy[maxIdx])
		originalCopy.remove(originalCopy[maxIdx])
	return cluster

print "Greedy 1-D Scores: "

scores = list()
avgscore = 0
count = 0
while count < n/k:
	count+=1
	c1 = clusterOriginal(originalCopy, len(originalCopy), k, divFact)
	x = computeTotalScore(c1, divFact)
	avgscore += x
	scores.append(x)
	#print "Cluster " + str(count) + " Score: " + str(x)+", Size: "+str(len(c1))

avgscore = avgscore/(n/k)
var = 0
for i in scores:
	var += math.pow(avgscore - i, 2)
stdev = math.sqrt(var)
print "Average Score: "+str(avgscore)
print "Standard Deviation: "+str(stdev)
print

########################################
#			Random Clustering 		   #
#			for Comparison 			   #
#									   #
########################################

randomClustering = list()
while len(randomClustering) < p:
	cluster1 = list()
	while len(cluster1) < k:
		rndnum = rand.randint(0, len(randomCopy)-1)
		next = randomCopy[rndnum]
		randomCopy.remove(randomCopy[rndnum])
		cluster1.append(next)
	randomClustering.append(cluster1)

print "Random Scores: "

scores = list()
avgscore = 0
for c1 in randomClustering:
	x = computeTotalScore(c1, divFact)
	avgscore += x
	scores.append(x)
	#print "Cluster " + str(count) + " Score: " + str(x)+", Size: "+str(len(c1))

avgscore = avgscore/(n/k)
var = 0
for i in scores:
	var += math.pow(avgscore - i, 2)
stdev = math.sqrt(var)
print "Average Score: "+str(avgscore)
print "Standard Deviation: "+str(stdev)
print

