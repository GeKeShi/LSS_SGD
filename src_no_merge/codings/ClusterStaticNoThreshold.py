# -*- coding: utf-8 -*
import numpy as np
import random
from scipy import stats
from Pair import *
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2

class ClusterStaticNoThreshold(object):


	def __init__(self, _numCluster=0, _maxIters=1000):
		self.totalGroup = _numCluster
		self.maxIters = _maxIters
		# self.threshold4Maximum=0
		# self.threshold4Maximum0=0
		self.centroids = [0]*_numCluster
		self.pctThreshold = 95

	@staticmethod
	def WhetherAdd2ClusterTrace(r, N, sampled):
		if(r <= (sampled+0.0) / N):
			return True
		else:
			return False

	def entropy_bin(self, points):
		fq = stats.relfreq(points, numbins=5, defaultreallimits=(np.amin(points), np.amax(points)))
		# print('entropy frequency', fq.frequency)
		return stats.entropy(fq.frequency, base=2)

	def entropy(self, points):
		fq = stats.itemfreq(points)[:, 1] / points.size
		return stats.entropy(fq, base=2)


	def selectThreshold(self, points):
		pctVal = np.percentile(points, self.pctThreshold)
		entropyVal = self.entropy_bin(np.extract(np.abs(points)>abs(pctVal), points))
		return [pctVal, 3*entropyVal]
	# /**
	# 	 * init cluster, return centroids
	# 	 * @param points
	# 	 * @param bufferedWriter 
	# 	 */
	def KPPCluster(self, points, bufferedWriter=None):	
		# /**
		#  * remember the ratio of each cluster
		#  */
		# //sizeRatio = new ArrayList<Double>(totalGroup);
		
		# long t1 = System.currentTimeMillis();
		# //threshold to do
		if np.amax(points)>0:
			pct= self.selectThreshold(points)
			self.threshold4Maximum =pct[0]
			pctEntropy=pct[1]
			
			# long t2 = System.currentTimeMillis();
			
			# LOG.info("KMeansSelectPct: "+(t2-t1) + " "+threshold4Maximum+", entropy: "+pctEntropy+"\n");
			# if(bufferedWriter!=null){
			# try {
			# 	bufferedWriter.write("KMeansSelectPct: "+(t2-t1) + " "+threshold4Maximum+"\n");
			# 	bufferedWriter.flush();
			# } catch (IOException e) {
			# 	// TODO Auto-generated catch block
			# 	e.printStackTrace();
			# }}
			
			# t1 = System.currentTimeMillis();
			clusterInput = np.extract(points < self.threshold4Maximum, points)
				# t2 = System.currentTimeMillis();
			#  // LOG.info("$: "+threshold4Maximum+", "+clusterInput.size);
				# LOG.info("KMeansSetPrepare: "+(t2-t1)+"\n");
			#   if(bufferedWriter!=null){
			#   try {
			# 	bufferedWriter.write("KMeansSetPrepare: "+(t2-t1)+"\n");
			# 	bufferedWriter.flush();
			# } catch (IOException e) {
			# 	// TODO Auto-generated catch block
			# 	e.printStackTrace();
			# }}
				
			# // initialize a new clustering algorithm. 
		#  // we use KMeans++ with  clusters and x iterations maximum.
		#  // we did not specify a distance measure; the default (euclidean distance) is used.
			kmeansClusterNum = self.totalGroup-1
				
				# //double[] centers= new double[kmeansClusterNum];
				# //ArrayList<Double> center = new ArrayList<Double>(kmeansClusterNum);
			center = []
			totalPoints = float(points.size)
			clusterPoints = clusterInput.size
			groupPercent = (totalPoints-clusterPoints)/totalPoints
			# //add
			center.append(Pair(groupPercent,float(self.threshold4Maximum), pctEntropy))
		else:
			self.pctThreshold = 5
			pct = self.selectThreshold(points)
			self.threshold4Maximum = pct[0]
			pctEntropy = pct[1]
			
			# long t2 = System.currentTimeMillis();
			
			# LOG.info("KMeansSelectPct: "+(t2-t1) + " "+threshold4Maximum+", entropy: "+pctEntropy+"\n");
			# if(bufferedWriter!=null){
			# try {
			# 	bufferedWriter.write("KMeansSelectPct: "+(t2-t1) + " "+threshold4Maximum+"\n");
			# 	bufferedWriter.flush();
			# } catch (IOException e) {
			# 	// TODO Auto-generated catch block
			# 	e.printStackTrace();
			# }}
			
			# t1 = System.currentTimeMillis();
			clusterInput = np.extract(points > self.threshold4Maximum, points)
				# t2 = System.currentTimeMillis();
			#  // LOG.info("$: "+threshold4Maximum+", "+clusterInput.size);
				# LOG.info("KMeansSetPrepare: "+(t2-t1)+"\n");
			#   if(bufferedWriter!=null){
			#   try {
			# 	bufferedWriter.write("KMeansSetPrepare: "+(t2-t1)+"\n");
			# 	bufferedWriter.flush();
			# } catch (IOException e) {
			# 	// TODO Auto-generated catch block
			# 	e.printStackTrace();
			# }}
				
			# // initialize a new clustering algorithm. 
		#  // we use KMeans++ with  clusters and x iterations maximum.
		#  // we did not specify a distance measure; the default (euclidean distance) is used.
			kmeansClusterNum = self.totalGroup-1
				
				# //double[] centers= new double[kmeansClusterNum];
				# //ArrayList<Double> center = new ArrayList<Double>(kmeansClusterNum);
			center = []
			totalPoints = float(points.size)
			clusterPoints = clusterInput.size
			groupPercent = (totalPoints-clusterPoints)/totalPoints
			# //add
			center.append(Pair(groupPercent,float(self.threshold4Maximum), pctEntropy))
		# //LOG.info("center: "+ center[0].value);
		
		# ManhattanDistance md = new ManhattanDistance();

		# kmeans = KMeans(n_clusters = kmeansClusterNum).fit(clusterInput.reshape(-1,1))
		# labels = kmeans.labels_.astype(np.int)
		# centerV = kmeans.cluster_centers_.reshape(-1)

		centerV, labels = kmeans2(data=clusterInput.reshape(-1,1), k=kmeansClusterNum)
		# print('centerv shape{}'.format(centerV.shape))
		centerV.reshape(-1)
		# print(centerV)
		clusterResults = []

		# to do np.count()
		for i in range(kmeansClusterNum):
			tmp_clusterresult = np.extract(labels == i, clusterInput)
			clusterResults.append(tmp_clusterresult)
			
			# print('tmp clusterResults{}, max {}, min {}, mean {}, error rate {}', tmp_clusterresult,tmp_clusterresult.max(),tmp_clusterresult.min(), tmp_clusterresult.mean(), np.abs((tmp_clusterresult.max()-tmp_clusterresult.min())/tmp_clusterresult.min()))
		# centerV = kmeans.cluster_centers_

		# KMeansPlusPlusClusterer<oneDimData> clusterer = new KMeansPlusPlusClusterer<oneDimData>(kmeansClusterNum,maxIters,md); 
		# //seed
		# RandomGenerator rg = clusterer.getRandomGenerator();
		# rg.setSeed(System.currentTimeMillis());
		
		# LOG.info("Cluster start! "+clusterInput.size);
		# t1 = System.currentTimeMillis();
		# //compute cluster
		# List<CentroidCluster<oneDimData>> clusterResults = clusterer.cluster(clusterInput);
		# LOG.info("Cluster completed!");
		for i in range(kmeansClusterNum):
			# //center
				# //cluster centers to do
			center.append(Pair(float(clusterResults[i].size)/totalPoints,
					centerV[i],self.entropy(clusterResults[i])))
			# print('Pair:',float(clusterResults[i].size)/totalPoints, centerV[i], self.entropy(clusterResults[i]))
				# //LOG.info("center: "+ centerV.getPoint()[0]);
				# //all nodes
	# //		    	for(oneDimData oneD: clusterResults.get(i).getPoints()){
	# //		    		LOG.info(POut.toString(oneD.getPoint()));
	# //		    	}
	# 		}
			# //sort the array to do
		sorted_center = sorted(center, key=lambda center_pair: center_pair.value)

		# 	t2 = System.currentTimeMillis();
			
		# 	LOG.info("KMeansDelay: "+(t2-t1)+"\n");
		# 	if(bufferedWriter!=null){ 
		# 	try {
		# 	bufferedWriter.write("KMeansDelay: "+(t2-t1)+"\n");
		# 	bufferedWriter.flush();
		# } catch (IOException e) {
		# 	// TODO Auto-generated catch block
		# 	e.printStackTrace();
		# }}
			
		return sorted_center
