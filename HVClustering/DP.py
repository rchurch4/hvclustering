# MDClusterRanking CLASS
# mdclusterranking.py
#
# contains:
#	- dynamic programming algorithm for clustering
#		by 1 dimension optimally
#	- use cluster1D() to create a clustering of all n points
#		over 1 dimension
#
# 1-D High Variance Clustering
# Dynamic Programming Optimal Clustering Algorithm
#
# Rob Churchill

import numpy as np
import math
import copy as cp

#def algorithm(dataset, p) compute best cluster of k students
def DP (d1, size, x, p): 
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

def cluster1D (dataset, size, x, p):
	d1 = cp.deepcopy(dataset)
	clusterInst = list()
	clusterAvg = list()
	#create n/k clusters of k points each
	count = 0
	while size > p:
		count+=1
		c1 = DP(d1, len(d1), x, p)
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