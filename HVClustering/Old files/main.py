# MAIN CLASS
# main.py
#
# contains:
#	- code to bring everything together
#	- algorithm and method for clustering
#
# 1-D High Variance Clustering
# Dynamic Programming Algorithm
#
# Rob Churchill

#import cluster
import random as rand
import numpy as np
import math

#def algorithm(dataset, k) compute best cluster of k students
def clustering (dataset, size, k): 
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
				newDist += math.fabs(cluster[clusterIdx][1] - dataset[index][1])
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

################################################################################
#   EDIT THESE VALUES TO CHANGE THE SIZE OF THE DATASET AND SIZE OF CLUSTERS   #
################################################################################
n = 30
k = 6
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
	intelligence = int(rand.normalvariate(50, 20)) #random number, 0-100
	nextStudent = (nextId, intelligence)
	dataset.append(nextStudent)
	nextId += 1

#sort the dataset by intelligence
dataset = sorted(dataset, key=lambda student: student[1])

#create n/k clusters of k points each
count = 0
while size > k-1:
	count+=1
	c1 = clustering(dataset, len(dataset), k)
	print "Cluster " + str(count) + ": " + str(c1)
	print "Score: " + str(computeScore(c1))
	print
	size = len(dataset)


	