# DYNAMIC CLASS
# dynamic.py
#
# contains:
#	- dynamic algorithm and method for clustering
#
# 1-D High Variance Clustering
# Dynamic Optimal Clustering Algorithm
#
# Rob Churchill

#import cluster
import random as rand
import numpy as np
import math

######## NOTE: Whenever you see dataset[x][1], that just means it's getting the intelligence value of student x

def clustering (dataset, size, p):
	

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
n = 10
p = 5
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
while size > p-1:
	count+=1
	c1 = clustering(dataset, len(dataset), p)
	print "Cluster " + str(count) + ": " + str(c1)
	#print "Score: " + str(computeScore(c1))
	print
	size = len(dataset)
	#stops after 1 iteration for testing purposes
	size = p-1
