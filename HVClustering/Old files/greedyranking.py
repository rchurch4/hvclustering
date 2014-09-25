# GREEDYRANKING CLASS
# greedyranking.py
#
# contains:
#	- algorithm and method for clustering
#	- algorithm to create better final clusters
#		through ranking 1-D clusters
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

#def algorithm(dataset, k) compute best cluster of k students
def clustering (dataset, size, k, x): 
	#create empty cluster
	cluster = list()
	#add smartest person in dataset to start
	cluster.append(dataset[len(dataset)-1])
	dataset.remove(dataset[len(dataset)-1])
	while len(cluster) < k:
		#index and score of best point so far
		maxDist = 0
		maxIdx = 0
		#index in dataset
		index = 0
		while index < len(dataset):
			newDist = 0
			clusterIdx = 0
			#get distance from point dataset[index] to all points in cluster
			while clusterIdx < len(cluster):
				a = math.fabs(cluster[clusterIdx][x] - dataset[index][x])
				print
				print "A"
				print a
				print
				newDist += a
				clusterIdx+=1

			#if its distance is higher than all others, change maxDist and maxIdx to reflect that
			if ( newDist > maxDist ):
				maxDist = newDist
				maxIdx = index
			index += 1
		#add the point with highest avg distance to the cluster, remove from dataset
		cluster.append(dataset[maxIdx])
		dataset.remove(dataset[maxIdx])

	return cluster
	#set base cases for k = 0, 1 & size = 0, 1
	#start the recursive function

#computes the final average distance of a cluster
def computeScore (cluster):
	score = 0
	index = 0
	for i in cluster:
		currentIdx = index + 1
		while currentIdx < len(cluster):
			score += math.fabs(i[1] - cluster[currentIdx][1])
			currentIdx += 1

	return score/len(cluster)	

# def cluster1D (dataset, size, k, x):
# 	d1 = dataset
# 	clusterInst = list()
# 	#create n/k clusters of k points each
# 	count = 0
# 	while size > k-1:
# 		count+=1
# 		c1 = clustering(d1, len(dataset), k, x)
# 		print "Cluster " + str(count) + ": " + str(c1)
# 		print "Score: " + str(computeScore(c1))
# 		print
# 		clusterInst.append(c1)
# 		size = len(dataset)

# 	return clusterInst

################################################################################
#   EDIT THESE VALUES TO CHANGE THE SIZE OF THE DATASET AND SIZE OF CLUSTERS   #
################################################################################
n = 100
k = 5
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
	math = int(rand.normalvariate(50, 20)) #random number, 0-100
	english = int(rand.normalvariate(50, 20)) #random number, 0-100
	history = int(rand.normalvariate(50, 20)) #random number, 0-100
	nextStudent = (nextId, math, english, history)
	dataset.append(nextStudent)
	nextId += 1

#sort the dataset by intelligence
dataset = sorted(dataset, key=lambda student: student[1])

d1 = dataset
size = len(d1)
d1Clusters = list()
#create n/k clusters of k points each
count = 0
while size > k-1:
	count+=1
	c1 = clustering(d1, len(dataset), k, 1)
	print "Cluster " + str(count) + ": " + str(c1)
	print "Score: " + str(computeScore(c1))
	print
	d1Clusters.append(c1)
	size = len(d1)


dataset = sorted(dataset, key=lambda student: student[2])

d1 = dataset
size = len(d1)
d2Clusters = list()
#create n/k clusters of k points each
count = 0
while size > k-1:
	count+=1
	c1 = clustering(d1, len(dataset), k, 2)
	print "Cluster " + str(count) + ": " + str(c1)
	print "Score: " + str(computeScore(c1))
	print
	d2Clusters.append(c1)
	size = len(d1)

dataset = sorted(dataset, key=lambda student: student[3])

d1 = dataset
size = len(d1)
d3Clusters = list()
#create n/k clusters of k points each
count = 0
while size > k-1:
	count+=1
	c1 = clustering(d1, len(dataset), k, 3)
	print "Cluster " + str(count) + ": " + str(c1)
	print "Score: " + str(computeScore(c1))
	print
	d3Clusters.append(c1)
	size = len(d1)

clusters = [d1Clusters, d2Clusters, d3Clusters]

#empty matrix [n, n]
F = np.zeros(n, n)

#for each 1-d clustering
for y in clusters:
	idx = 0
	#take each cluster, with factor higher for earlier clusters
	while idx < len(y):
		c1 = y[idx]
		factor = len(y) - idx
		#and for each pair in the cluster
		i1 = 0
		while i1 < len(c1)-1:
			i2 = i1+1
			while i2 < len(c1):
				#add the factor of the cluster to their partnership F[x, q]
				F[c1[i1], c1[i2]] += factor
				i2+=1
			i1+=1

print F
#for n/k clusters, take the highest pair and place them in a cluster together, s.t.
#neither point is already in a cluster
numClusters = n/k
print np.argmax(F)



	