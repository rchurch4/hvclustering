# GR CLASS
# GR.py
#
# contains:
#	- greedy multi-dimensional algorithm
#
# 1-D High Variance Clustering
# Greedy Approximation Clustering Algorithm
#
# Rob Churchill

import math

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