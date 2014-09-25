
# DPTEST CLASS
# DPtest.py
#
# contains:
#	- dynamic programming algorithm and testing code
#
# 1-D High Variance Clustering
# Dynamic Programming Clustering Algorithm
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
p = 200
normMean = 50
normStdev = 20
resultsFile = open('results.csv', 'a')
################################################################################
# ^ EDIT THESE VALUES TO CHANGE THE SIZE OF THE DATASET AND SIZE OF CLUSTERS ^ #
################################################################################

#def algorithm(dataset, k) compute best cluster of k students
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
		print "Cluster " + str(count) + ": " + str(len(c1[0]))
		print "Score: " + str(computeScore(c1[0], x)) + ", OPT = "+str(c1[1])
		print "Avg Dist: "+str(computeAvgDistance(c1[0]))
		print
		clusterInst.append(c1[0])

		size = len(d1)
	
	if size > 0:
		print "Cluster " + str(count+1) + ": " + str(len(d1))
		print "Score: " + str(computeScore(d1, x))
		print "Avg Dist: "+str(computeAvgDistance(d1))
		print
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

#create dataset
dataset = list()
nextId = 1
size = 0

print "N = "+str(n)+", p = "+str(p)

#create dataset according to a normal distribution with mean 50 and stdev 20
while size < n:
	size += 1
	meth = int(rand.normalvariate(normMean, normStdev)) #random number, 0-100
	english = int(rand.normalvariate(normMean, normStdev)) #random number, 0-100
	history = int(rand.normalvariate(normMean, normStdev)) #random number, 0-100
	nextStudent = (nextId, meth, english, history)
	dataset.append(nextStudent)
	nextId += 1

size = len(dataset)

dataset = sorted(dataset, key=lambda student: student[1])
l1 = cluster1D(dataset, size, 1)

#dataset = sorted(dataset, key=lambda student: student[2])
#l2 = cluster1D(dataset, size, 2)

#dataset = sorted(dataset, key=lambda student: student[3])
#l3 = cluster1D(dataset, size, 3)
