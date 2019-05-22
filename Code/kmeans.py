# Author: Samuel Heersink
# November 2018
import matplotlib.pyplot as plot;
import numpy;
import datetime;
import csv;
from sklearn.cluster import KMeans
from sklearn import metrics

# constants


# region set info compilation
# The SetInfo class holds all the information about the datasets
class SetInfo:
    def __init__(self, l, tL, kk):
        self.location = l
        self.truthLocation = tL 
        self.k = kk

# A is three two dimensional sets with varying points and Ks
a1Info = SetInfo("Data/A/a1.txt", "Data/A/a1-gt.txt", 20)
a2Info = SetInfo("Data/A/a2.txt", "Data/A/a2-gt.txt", 35)
a3Info = SetInfo("Data/A/a3.txt", "Data/A/a3-gt.txt", 50)
aData = [a1Info, a2Info, a3Info]

# Birch is Three sets, each 100k points in two dimensions, in 100 clusters
birchNames = ["b1", "b2", "b3"]
birchData = []
for name in birchNames:
    birchData.append(SetInfo("Data/Birch/" + name + ".txt", "Data/Birch/" + name + "-gt.txt", 100))

# Dim is 6 sets in varying dimensions, with 16 clusters in each. 1024 points in each.
dimNames = ["dim032", "dim064", "dim128", "dim256", "dim512", "dim1024"]
dimData = []
for name in dimNames:
    dimData.append(SetInfo("Data/Dim/" + name + ".txt", "Data/Dim/" + name + "-gt.txt", 16))

# G2 is 110 sets in 2 to 1024 dimensions, each with 2048 points in two clusters.
g2Data = []
i = 0
while i < 11:
    j = 10
    while j < 101:
        name = "g2-" + str(2**i) + "-" + str(j)
        g2Data.append(SetInfo("Data/G2/" + name + ".txt", "Data/G2/" + name + "-gt.txt", 2))
        j += 10
    i += 1

# S is three different sets in two dimensions, each with 15 clusters of 5000 points.
sNames = ["s1", "s2", "s3"]
sData = []
for name in sNames:
    sData.append(SetInfo("Data/S/" + name + ".txt", "Data/S/" + name + "-gt.txt", 15))
#endregion

# The algorithm for performing BKM on a dataset, returns a K-Clustering of the dataset by splitting hierarchically.
# If splitWorst is set to true then the algorithm will split the cluster with the highest mean inertia.
# If it is not set, the largest cluster will be split instead.
# If graph is set to true, the method will also display a graph of the results.
# Returns a kmeans object for manipulation.
def bisectingKMeans (setInfo, splitWorst = False, graph = False):
    numClusters = 1
    data = numpy.loadtxt(setInfo.location)
    if len(data.shape) < 2:
        data = data.reshape(-1, 1)
    #Initialize the collection of clusters to one giant cluster of all the data
    clusters = [data]
    centers = []
    inertias = []

    #Loop until we reach K clusters
    while len(clusters) < setInfo.k:
        #Get the worst or largest clustering so that we can split it
        splitIndex = 0
        if splitWorst:
            # Get the cluster with the maximum inertia
            if inertias:
                i = 1
                while i < len(inertias):
                    if inertias[i] > inertias[splitIndex]:
                        splitIndex = i
                    i += 1
                # Remove the inertia measure from the list                
                del inertias[splitIndex]
        else:
            # Get the largest cluster
            i = 1
            while i < len(clusters):
                if clusters[i].shape[0] > clusters[splitIndex].shape[0]:
                    splitIndex = i
                i += 1
        #end if
        splitCluster = clusters[splitIndex]

        #Remove the chosen cluster and its center from their lists
        del clusters[splitIndex]
        if centers:
            del centers[splitIndex]

        #Perform K-Means on the chosen cluster
        if len(splitCluster.shape) < 2:
            splitCluster = splitCluster.reshape(-1, 1)
        kmeans = KMeans(n_clusters = 2, init = "random", n_init = 10, n_jobs = -1, algorithm="full")
        kmeans.fit(splitCluster)
        clusterAssignment = kmeans.labels_

        #Add the cluster centers to the list
        centers.append(kmeans.cluster_centers_[0])
        centers.append(kmeans.cluster_centers_[1])
        
        # Split the data into two based on the clustering
        firstHalf = []
        secondHalf = []
        i = 0
        while i < len(clusterAssignment):
            if clusterAssignment[i] == 0:
                firstHalf.append(splitCluster[i])
            else:
                secondHalf.append(splitCluster[i])
            i+=1
 
        # Add the two clusters to the list
        firstCluster = numpy.array(firstHalf)
        secondCluster = numpy.array(secondHalf)       
        clusters.append(firstCluster)
        clusters.append(secondCluster)

        # Calculate the mean inertia of each cluster if we are splitting on the worst one
        if splitWorst:
            inertiaFirstCluster = KMeans(n_clusters=1, n_init = 1, n_jobs = -1, algorithm = "full").fit(firstCluster).inertia_ / firstCluster.shape[0]
            inertiaSecondCluster = KMeans(n_clusters=1, n_init = 1, n_jobs = -1, algorithm = "full").fit(secondCluster).inertia_ / secondCluster.shape[0]
            inertias.append(inertiaFirstCluster)
            inertias.append(inertiaSecondCluster)
    # end while

    # Combine all the clusters once more by performing kmeans seeded with the known centers
    clusterCentersArray = numpy.array(centers)
    result = KMeans(n_clusters = setInfo.k, init = clusterCentersArray, n_jobs = -1, n_init = 1, algorithm = "full").fit(data)

    #Graphing
    if graph:
        clusterID = result.predict(data)
        plot.scatter(data[:, 0], data[:, 1], c=clusterID, s=5, cmap='gnuplot2')
        plot.show()
    
    return result

# This method evaluates a given KMeans clustering on specific data according to a few different metrics.
# Returns: list(inertia, adjusted rand index, mutual information score, v-measure, fowlkes-mallows score)
def evaluateResults (info, clustering):
    resultsList = []
    predictedLabels = clustering.labels_
    # Get the optimal labelling by calling k-means one more time with the optimal centers
    trueCenters = numpy.loadtxt(info.truthLocation)
    data = numpy.loadtxt(info.location)
    if len(data.shape) < 2:
            data = data.reshape(-1, 1)
            trueCenters = trueCenters.reshape(-1, 1)
    trueLabels = KMeans(n_clusters = info.k, init = trueCenters, n_init = 1, n_jobs = -1, algorithm = "full").fit(data).labels_
    #inertia
    resultsList.append(clustering.inertia_)
    #ARI
    resultsList.append(metrics.adjusted_rand_score(trueLabels, predictedLabels))
    #MI
    resultsList.append(metrics.mutual_info_score(trueLabels, predictedLabels))
    #VM
    resultsList.append(metrics.v_measure_score(trueLabels, predictedLabels))
    #FM
    resultsList.append(metrics.fowlkes_mallows_score(trueLabels, predictedLabels))
    return resultsList

# The function that runs the tests. It runs three tests for each data set (one per algorithm) and saves a number of features for each test.
def runTest (infoList, fileName):
    writeFile = open("Results/" + fileName + ".csv", 'w+', newline = '')
    writer = csv.writer(writeFile)
    # Write the header
    writer.writerow(["Name", "Algorithm", "Time Elapsed", "Inertia", "Adjusted Rand Index", "Mutual Information Score", "V-Measure", "Folkes-Mallows Score"])
    for info in infoList:
        #Get the set's name (the portion that appears in the filename before .txt)
        name = info.location.split('/')[2].split('.')[0]
        print("Beginning tests on " + name)

        # Get results for regular K-Means
        startTime = datetime.datetime.now()
        data = numpy.loadtxt(info.location)
        if len(data.shape) < 2:
            data = data.reshape(-1, 1)
        kmeans = KMeans(n_clusters = info.k, init = "random", n_jobs = -1, n_init = 10, algorithm="full").fit(data)
        elapsedTime = (datetime.datetime.now() - startTime)
        resultsList = evaluateResults(info, kmeans)
        fullResultsList = [name, "K-Means", str(elapsedTime)] + resultsList
        writer.writerow(fullResultsList)
        print("Finished K-Means on " + name)

        # Get results for BKM split on largest cluster
        startTime = datetime.datetime.now()
        bkmLargest = bisectingKMeans(info)
        elapsedTime = (datetime.datetime.now() - startTime)
        resultsList = evaluateResults(info, bkmLargest)
        fullResultsList = [name, "Bisecting K-Means (largest split)", str(elapsedTime)] + resultsList
        writer.writerow(fullResultsList)
        print("Finished BKM-Largest on " + name)

        #Get results for BKM split on worst cluster
        startTime = datetime.datetime.now()
        bkmWorst = bisectingKMeans(info, True)
        elapsedTime = (datetime.datetime.now() - startTime)
        resultsList = evaluateResults(info, bkmWorst)
        fullResultsList = [name, "Bisecting K-Means (worst split)", str(elapsedTime)] + resultsList
        writer.writerow(fullResultsList)
        print("Finished BKM-Worst on " + name)
    #end for
    writeFile.close()

runTest(birchData, "birchResultsRandomSeeds")