# MDClusterRanking CLASS
# mdclusterranking.py
#
# contains:
#	- algorithm and method for clustering
#	- algorithm to create better final clusters
#		through ranking 1-D clusters
#	- clusters through draft-like process (round robin) & aggregation
#	- MAX AVG DISTANCE
#	- any number of dimensions > 1
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
import DP
import GR

################################################################################
#   EDIT THESE VALUES TO CHANGE THE SIZE OF THE DATASET AND SIZE OF CLUSTERS   #
################################################################################
n = 1000 # number of points in the set
p = 200 # total numbers points per cluster
l = n/p # total number of clusters
d = 3 # number of dimensions in each point
normMean = 50
normStdev = 20
resultsFile = open('dimensionresults.csv', 'a')
################################################################################
# ^ EDIT THESE VALUES TO CHANGE THE SIZE OF THE DATASET AND SIZE OF CLUSTERS ^ #
################################################################################

def rank(clusters, n, p):
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

	return F

def draft(F, dataset, n, p, l):
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
	return clusters

def aggregate(F, dataset, n, p, l):
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
	return clusters

def printScores(resultsFile, clusters, l):
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
	#print "Average Score: "+str(avgscore)
	resultsFile.write(str(avgscore)+",")
	#print "Standard Deviation: "+str(stdev)
	resultsFile.write(str(stdev)+",")
	#print "Average Distance: "+str(avgdistance)
	resultsFile.write(str(avgdistance)+",")
	#print

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

for repeat in range(0,20):
	#create dataset
	dataset = list()
	nextId = 0
	size = 0

	print "N = "+str(n)+", P = "+str(p)

	#create dataset according to a normal distribution with mean 50 and stdev 20
	while size < n:
		size += 1
		attributes = list()
		attributes.append(nextId)
		while len(attributes) < d+1:
			meth = int(rand.normalvariate(normMean, normStdev)) #random number, 0-100
			attributes.append(meth)

		nextStudent = tuple(attributes)
		dataset.append(nextStudent)
		nextId += 1

	#sort the dataset by intelligence
	randomCopy = cp.deepcopy(dataset)
	dataset = sorted(dataset, key=lambda student: student[1])

	originalCopy = cp.deepcopy(dataset)
	l1 = DP.cluster1D(dataset, size, 1, p)

	clusters = list()
	clusters.append(l1)

	dim = 2
	while dim <= d:
		dataset = sorted(dataset, key=lambda student: student[dim])
		l2 = DP.cluster1D(dataset, size, dim, p)
		clusters.append(l2)
		dim += 1


	#####################################
	#		   Cluster Ranking 			#
	#									#
	#####################################

	# matrix of point relationship ranks
	F = rank(clusters, n, p)
	gryffindor = cp.deepcopy(F)

	##############################################
	# 	 	DP Draft Clustering algorithm   	 #
	# 											 #
	# 		  The best algorithm so far			 #
	##############################################

	# resort dataset by ID number for fast withdrawal
	dataset = sorted(dataset, key=lambda student: student[0])

	# create approximation clusters via draft algorithm
	clusters = draft(F, dataset, n, p, l)

	# for i in clusters:
	# 	print i
	print "DP Draft Scores: "

	printScores(resultsFile, clusters, l)

	##############################################
	# 	 DP Aggregate Clustering algorithm   	 #
	# 											 #
	# 		    For Comparison's sake			 #
	##############################################

	dataset = sorted(dataset, key=lambda student: student[0])

	F = gryffindor

	# create clusters by cluster aggregation algorithm
	clusters = aggregate(F, dataset, n, p, l)

	# for i in clusters:
	# 	print i
	print "DP Aggregate Scores: "

	printScores(resultsFile, clusters, l)

	##############################################
	#  Original Greedy M-D Clustering Algorithm  #
	# 											 #
	# 		    For Comparison's sake			 #
	##############################################

	print "Greedy M-D Scores: "

	scores = list()
	avgscore = 0
	avgdistance = 0
	count = 0
	while count < l:
		count+=1
		c1 = GR.clusterOriginal(originalCopy, len(originalCopy), p)
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
	#print "Average Score: "+str(avgscore)
	resultsFile.write(str(avgscore)+",")
	#print "Standard Deviation: "+str(stdev)
	resultsFile.write(str(stdev)+",")
	#print "Average Distance: "+str(avgdistance)
	resultsFile.write(str(avgdistance)+",")
	#print

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
	#print "Average Score: "+str(avgscore)
	resultsFile.write(str(avgscore)+",")
	#print "Standard Deviation: "+str(stdev)
	resultsFile.write(str(stdev)+",")
	#print "Average Distance: "+str(avgdistance)
	resultsFile.write(str(avgdistance)+'\n')
	#print

resultsFile.write("End Test of d = "+str(d)+'\n')

