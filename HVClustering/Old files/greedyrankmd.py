
# GREEDYRANK CLASS
# greedyrank.py
#
# contains:
#	- algorithm and method for clustering
#	- algorithm to create better final clusters
#		through ranking 1-D clusters
#	- original greedy algorithm for comparison's sake
#
# MULTI-D High Variance Clustering
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
def clustering (d1, size, k, x): 
	#create empty cluster
	cluster = list()
	#add smartest person in dataset to start
	cluster.append(d1[len(d1)-1])
	d1.remove(d1[len(d1)-1])
	while len(cluster) < k:
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

def computeTotalScore(cluster):
	score = 0
	index = 1
	while index < len(cluster[0]):
		score += computeScore(cluster, index)
		index += 1

	score = score/(3*(len(cluster[0])-1))
	return score

def getRep (cluster, x):
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
		return -100

	#computes the final average distance of a cluster
def computeScore (cluster, x):
	score = 0
	index = 0
	for i in cluster:
		currentIdx = index + 1
		while currentIdx < len(cluster):
			score += math.fabs(i[x] - cluster[currentIdx][x])
			currentIdx += 1

	return score/len(cluster)

def cluster1D (d1, size, k, x):
	d1 = cp.deepcopy(dataset)
	clusterInst = list()
	#create n/k clusters of k points each
	count = 0
	while size > k-1:
		count+=1
		c1 = clustering(d1, len(d1), k, x)
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
n = 50
k = 10
p = n/k
################################################################################
# ^ EDIT THESE VALUES TO CHANGE THE SIZE OF THE DATASET AND SIZE OF CLUSTERS ^ #
################################################################################
print "v2"

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
l1 = cluster1D(dataset, size, k, 1)
dataset = sorted(dataset, key=lambda student: student[2])
l2 = cluster1D(dataset, size, k, 2)
dataset = sorted(dataset, key=lambda student: student[3])
l3 = cluster1D(dataset, size, k, 3)
clusters = [l1, l2, l3]

#empty matrix [n, n]
F = np.zeros((n, n))

#for each 1-d clustering
for y in clusters:
	idx = 0
	#take each cluster, with factor higher for earlier clusters
	while idx < len(y):
		c1 = y[idx]
		factor = len(y) - 2*idx
		if factor < 0:
			factor = 0
		#and for each pair in the cluster
		i1 = 0
		while i1 < len(c1)-1:
			i2 = i1+1
			while i2 < len(c1):
				#add the factor of the cluster to their partnership F[x, q]
				F[c1[i1][0], c1[i2][0]] += factor
				F[c1[i2][0], c1[i1][0]] += factor
				i2+=1
			i1+=1
		idx+=1

#for n/k clusters, take the highest pair and place them in a cluster together, s.t.
#neither point is already in a cluster
numClusters = n/k

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
	x = computeTotalScore(i)
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
def clusterOriginal (originalCopy, size, k): 
	#create empty cluster
	cluster = list()
	#add smartest person in dataset to start
	cluster.append(originalCopy[len(originalCopy)-1])
	originalCopy.remove(originalCopy[len(originalCopy)-1])
	while len(cluster) < k:
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

	return cluster
	#set base cases for k = 0, 1 & size = 0, 1
	#start the recursive function

print "Greedy M-D Scores: "

scores = list()
avgscore = 0
count = 0
while count < n/k:
	count+=1
	c1 = clusterOriginal(originalCopy, len(originalCopy), k)
	x = computeTotalScore(c1)
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
	x = computeTotalScore(c1)
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


