# -*- coding: utf-8 -*

import numpy as np
from TrainCluster import *
from ClusterStaticNoThreshold import *
from LSSEntry import *
import sys
# import xxhash

class LSSSimplified(object):
    # /**
    #  * record sketch
    #  */
    # public ArrayList<LSSEntry[]> LSSTable = null;
    # LSSTable = None
    
    # /**
    #  * parameters
    #  */
    # public int clusterNum = -1;


    # //hash function
    # public LongHashFunction[] LongHashFunction4PosHash;
    # LongHashFunction4PosHash = None
    MinArraySize = 10
    maxIterations4Cluster = 1000
    # //boolean useDirect = false;

    # /**
    #  * constructor
    #  *
    #  * @param useDirectIndex
    #  * @param bufferedWriter
    #  * @param _n,            umber of flow entries
    #  * @param _c,            cluster number
    #  * @param _b,            bucket count,
    #  * @param _fp,           false positive of bloom filters
    #  * @param traces,        flow trace
    #  */
    def __init__(self, _n, _c,  _b, _clusterArrayChoiceMethod, **kwargs):
        # //expectedNumItems = _n;
        self.clusterCountPerSide = _c
        self.bucketCount = _b
        # //expectedFP = _fp;
        # //scaleFactors = _scaleFactor;
        self.LSSTable = []

        self.clusterArrayChoiceMethod = _clusterArrayChoiceMethod
        self.center_split = _c
        # if 'LSSTable' in kwargs.keys():
        #     self.LSSTable = kwargs['LSSTable']
        # else:
        #     self.LSSTable = None

        # if 'LongHashFunction4PosHash' in kwargs.keys():
        #     self.LongHashFunction4PosHash = kwargs['LongHashFunction4PosHash']
        # else:
        #     self.LongHashFunction4PosHash = None
        # self.LongHashFunction4PosHash = None

    # /**
    #  * train the cluster model
    #  *
    #  * @param traces
    #  * @return
    #  */
    def trainClusterAndSetArraySize(self, traces0):
        PositiveItems = TrainCluster.getPositive(traces0)
        # print(PositiveItems.size, traces0.size)
        positiveFrac = float(PositiveItems.size) / float(traces0.size)
        # print(PositiveItems.size, traces0.size, positiveFrac)
        negativeFrac = 1 - positiveFrac
        # print('negativeFrac:{}'.format(negativeFrac))
        # PositiveItems = np.sort(PositiveItems)

        if PositiveItems.size > 0:
            # //only keep cluster center
            positiveclusterCenters = np.zeros(self.clusterCountPerSide)
            positiveclusterdensity = np.zeros(self.clusterCountPerSide)
            positiveclusterentropy = np.zeros(self.clusterCountPerSide)


            # /**
            #  * init cluster center, cluster centroid
            #  */
            self.initClusterCenters(PositiveItems, positiveFrac, positiveclusterCenters, positiveclusterdensity, positiveclusterentropy)
            # print('positiveclusterCenters {}, positiveclusterdensity {}, positive entropy {}'.format(positiveclusterCenters, positiveclusterdensity, positiveclusterentropy))

        nonPositives = TrainCluster.getNonPositive(traces0)

        # nonPositives = np.sort(nonPositives)

        if nonPositives.size > 0:
            # //only keep cluster center
            nonpositiveclusterCenters = np.zeros(self.clusterCountPerSide)
            nonpositiveclusterdensity = np.zeros(self.clusterCountPerSide)
            nonpositiveclusterentropy = np.zeros(self.clusterCountPerSide)
            # /**
            #  * init cluster center, cluster centroid
            #  */
            self.initClusterCenters(nonPositives, negativeFrac, nonpositiveclusterCenters, nonpositiveclusterdensity, nonpositiveclusterentropy)
            # print('nonpositiveclusterCenters {}, nonpositiveclusterdensity {}, nopositive entropy {}'.format(nonpositiveclusterCenters, nonpositiveclusterdensity, nonpositiveclusterentropy))


        # # //merge list
        # # List<Double> list0 = new ArrayList<Double>();
        # list0 = []
        # # //one side
        # if nonpositiveclusterCenters != None:
        #     addAll(list0, nonpositiveclusterCenters)

        # # //the other
        # if positiveclusterCenters != None:
        #     addAll(list0, positiveclusterCenters)

        clusterCenters = np.concatenate((nonpositiveclusterCenters, positiveclusterCenters), axis=0)
        # //fix the cluster num
        clusterNum = len(clusterCenters)
        # print('cluster centers', clusterCenters)
        # //late binding
        # LongHashFunction4PosHash = new LongHashFunction[clusterNum];
        # self.LongHashFunction4PosHash = [xxhash.xxh32(seed=i) for i in range(clusterNum)]
        # for i in range(clusterNum):
        #     LongHashFunction4PosHash = LongHashFunction.xx(i);
        

        # /**
        #  * set up bucket array by desity
        #  */
        # LSSTable = new ArrayList<LSSEntry[]>(clusterNum);
        # self.LSSTable = []

        # List<Double> list2 = new ArrayList<Double>();
        # if (nonpositiveclusterdensity != null) {
        #     addAll(list2, nonpositiveclusterdensity);
        # }
        # if (positiveclusterdensity != null) {
        #     addAll(list2, positiveclusterdensity);
        # }
        # double[] clusterdensity = Doubles.toArray(list2);
        clusterdensity = np.concatenate((nonpositiveclusterdensity, positiveclusterdensity), axis=0)
        clusterentropy = np.concatenate((nonpositiveclusterentropy, positiveclusterentropy), axis=0)
        

        # List<Double> list3 = new ArrayList<Double>();
        # if (nonpositiveclusterentropy != null) {
        #     addAll(list3, nonpositiveclusterentropy);
        # }
        # if (positiveclusterentropy != null) {
        #     addAll(list3, positiveclusterentropy);
        # }
        # double[] clusterentropy = Doubles.toArray(list3);

        # LOGGER.info("Cluster initialized! " + clusterentropy.length + ", eachSide: " + this.clusterCountPerSide);
        totalSum = self.clusterTotalSum(clusterCenters, clusterdensity, clusterentropy, self.clusterArrayChoiceMethod)
        # LOGGER.info("totalSum: " + totalSum);
        finalLeft = self.bucketCount
        # print('clustercenters {}, clusterdensity {}, clusterentropy {}'.format(clusterCenters, clusterdensity, clusterentropy))
        # densityTotalSum = np.sum(clusterdensity)
        # for (int i = 0; i < clusterdensity.length; i++) {
        #     densityTotalSum += clusterdensity[i];
        # }

        for i in range(clusterNum):
            # //scaled by cluster size
            # //int arraySize = Math.round(bucketCount/clusterCount);
            # arraySize = int(max(LSSSimplified.MinArraySize, round(self.getArraySize(clusterCenters, clusterdensity, clusterentropy, i, self.clusterArrayChoiceMethod, totalSum) * self.bucketCount)))
            # if finalLeft - arraySize < 0:
            #     # //log.warn("exceed the expected bucket!");
            #     arraySize = finalLeft
            # arraySize = max(1, arraySize)
            # # //LOGGER.info("cluster: "+i+", "+arraySize);
            # # //new array
            # b = [LSSEntry() for i in range(arraySize)]
            # # print('array size', arraySize)
            # self.LSSTable.append(b)
            # # print(self.LSSTable)
            # # //tune the bucket size
            # finalLeft -= arraySize

            arraySize = int(max(LSSSimplified.MinArraySize, round(self.getArraySize(clusterCenters, clusterdensity, clusterentropy, i, self.clusterArrayChoiceMethod, totalSum))))
            arraySize = max(1, arraySize)
            b = [LSSEntry() for i in range(arraySize)]
            # print('array size', arraySize)
            self.LSSTable.append(b)
        return clusterCenters


    def initClusterCenters(self, traces, signFrac, clusterCenters, clusterdensity, clusterentropy):
        # /**
        #  * cluster
        #  */
        test = ClusterStaticNoThreshold(clusterCenters.size, LSSSimplified.maxIterations4Cluster)
        # to do ？
        centers = test.KPPCluster(traces)
        for i in range(len(centers)):
            clusterCenters[i] = centers[i].value
            clusterdensity[i] = centers[i].index * signFrac
            clusterentropy[i] = centers[i].entropyVal
        cluster_center_max =  (np.abs(clusterCenters)).max()
        density_sum = np.sum(clusterdensity)
        entropy_sum = np.sum(clusterentropy)
        # print('before norm {}'.format(clusterentropy))
        for i in range(len(centers)):
            # clusterCenters[i] = clusterCenters[i]/Center_sum
            clusterdensity[i] = clusterdensity[i]/density_sum
            clusterentropy[i] = clusterentropy[i]/entropy_sum
        # clusterCenters = clusterCenters/Center_sum
        # clusterdensity = clusterdensity/density_sum
        # clusterentropy = clusterentropy/entropy_sum
        # print('centersum {}, density sum {}, entropy sum {}, centers {}, density {}, entropy {}'.format(cluster_center_max, density_sum, entropy_sum,clusterCenters, clusterdensity, clusterentropy))

# /**
#      * append to the final
#      *
#      * @param list0
#      * @param nonpositiveclusterCenters
#      */
    def addAll(self, list0, nonpositiveclusterCenters):
        if list0 != None and nonpositiveclusterCenters != None:
            for i in range(nonpositiveclusterCenters.size):
                list0.append(nonpositiveclusterCenters[i])
# /**
#      * cluster sum
#      *
#      * @param clusterCenters
#      * @param clusterdensity
#      * @param clusterentropy
#      * @param choiceArray
#      * @return
#      */
    def clusterTotalSum(self, clusterCenters, clusterdensity, clusterentropy, choiceArray):
        cluster_center_max = (np.abs(clusterCenters)).max()
        f = 0
        for i in range(clusterCenters.size):
            if choiceArray == 1:
                # //entropy*center
                f += clusterentropy[i] * abs((clusterCenters[i])/cluster_center_max) * clusterdensity[i]
            elif choiceArray == 2:
                # //entropy*density
                f += clusterentropy[i] *  abs((clusterCenters[i])/cluster_center_max)
            elif choiceArray == 3:
                # //entropy
                f += clusterentropy[i]
            else:
                # //density
                f += clusterdensity[i]
        return f

    def getArraySize(self, clusterCenters, clusterdensity, clusterentropy, i, choiceArray, totalSum):
        cluster_center_max = (np.abs(clusterCenters)).max()
        if choiceArray == 1:
            # //entropy*center
            # print('getarrarysize:',clusterentropy[i] , clusterCenters[i] , clusterdensity[i], clusterentropy[i] * abs((clusterCenters[i])/cluster_center_max) * clusterdensity[i]/(totalSum))
            return clusterentropy[i] * abs((clusterCenters[i])/cluster_center_max) * clusterdensity[i] / (totalSum)
        elif choiceArray == 2:
            # //entropy*density
            return self.bucketCount*(clusterentropy[i] *  abs((5*clusterCenters[i])/cluster_center_max) / totalSum)

        elif choiceArray == 3:
            # //entropy
            bucket_number = [500000, 200,200, 500000]# //entropy
            return bucket_number[i]
        else:
            # //density
            return clusterdensity[i] / totalSum


    def groupInputKV(self, flowValue, clusterCenters):
        # line_index = -1
        # vat = float("inf")
        # for i in range(clusterCenters.size):
        #     v = abs(clusterCenters[i] - flowValue)
        #     # //closer and same symbol
        #     if clusterCenters[i] * flowValue >= 0:
        #         if v < vat:
        #             vat = v
        #             line_index = i
        # return index
        
        #binsearch
        rescenters = np.abs(clusterCenters) - abs(flowValue)
        if flowValue > 0:
            low = self.center_split
            high =clusterCenters.size - 1
            if rescenters[low] > 0:
                return low
            if rescenters[high] < 0:
                return high       
            while low < high:
                mid = int((low + high)/2)
                if rescenters[mid]<0:
                    if rescenters[mid+1]<0:
                        low = mid + 1
                    else:
                        index = mid if -(rescenters[mid]) < rescenters[mid+1] else mid+1
                        # if line_index != index:
                        #     print(flowValue)
                        return index
                elif rescenters[mid]>0:
                    if rescenters[mid-1]>0:
                        high = mid -1
                    else:
                        index = mid if rescenters[mid] < -(rescenters[mid-1]) else mid-1
                        # if line_index != index:
                        #     print(flowValue)
                        return index
                else:
                    # if line_index != mid:
                    #     print(flowValue)
                    return mid
        else:
            low = 0
            high =self.center_split - 1
            if rescenters[low] < 0:
                return low
            if rescenters[high] > 0:
                return high       
            while low < high:
                mid = int((low + high)/2)
                if rescenters[mid]<0:
                    if rescenters[mid-1]<0:
                        high = mid - 1
                    else:
                        index = mid if -(rescenters[mid]) < rescenters[mid-1] else mid-1
                        # if line_index != index:
                        #     print(flowValue)
                        return index
                elif rescenters[mid]>0:
                    if rescenters[mid+1]>0:
                        low = mid + 1
                    else:
                        index = mid if rescenters[mid] < -(rescenters[mid+1]) else mid+1
                        # if line_index != index:
                        #     print(flowValue)
                        return index
                else:
                    # if line_index != mid:
                    #     print(flowValue)
                    return mid
    # /**
    #  * helper function
    #  *
    #  * @param id byte
    #  * @param index
    #  * @return
    #  */
    @staticmethod
    def hashPos(id, index):
        longHash = hash(id)
        # LongHashFunction4PosHash[index].update(id)
        # longHash = LongHashFunction4PosHash[index].intdigest()
        # python2
        # return int(abs(longHash % sys.maxint))
        return int(abs(longHash % sys.maxsize))

    # /**
    #  * inset
    #  *
    #  * @param key byte
    #  * @param val
    #  * @param clusterCenters
    #  * @return
    #  */
    def insert(self, key, val, clusterCenters):
        clusterIndex = self.groupInputKV(val, clusterCenters)
        # clusterIndex = np.searchsorted(val, clusterCenters)
        if clusterIndex < 0:
            return -1
        # //find an array containing the cluster of this value
        array = self.LSSTable[clusterIndex]

        # //LOGGER.info("$: "+clusterIndex+", "+array.length+", val="+val);

        # //find position in an array in the LSSTable
        # pos = LSSSimplified.hashPos(key, clusterIndex, self.LongHashFunction4PosHash) % (len(array))
        pos = LSSSimplified.hashPos(key, clusterIndex) % (len(array))
        # //assume unique id;
        array[pos].update(val)
        # //store

        # LOGGER.debug("index: " + clusterIndex + ", centroid: " + clusterCenters[clusterIndex] + ", val: " + val + ", bkt: " + array[pos].getAvgEstimator() + ", @: " + array[pos].getCounter());

        return clusterIndex
    
    def get_lsstable(self):
        LSSTable_val = []
        for LSSTable_item in self.LSSTable:
            LSSTable_val_item = []
            for item in LSSTable_item:
                # print('(item.sum){}/(item.counter){}'.format(item.sum, item.counter))
                LSSTable_val_item.append((item.sum)/(item.counter) if item.counter >=1 else 0)
            # print('LSStable_item: {}'.format(LSSTable_val_item))
            LSSTable_val.append(LSSTable_val_item)
        return LSSTable_val
    
    # key: byte
    @staticmethod
    def query(key, clusterIndex, coded_data):
        QueryTable = coded_data
        array = QueryTable[clusterIndex]
        # //find position
        pos = LSSSimplified.hashPos(key, clusterIndex) % (len(array))
        avgVal = array[pos].getAvgEstimator()
        return avgVal

    # key: byte
    @staticmethod
    def query_val(key, clusterIndex, coded_data):
        QueryTable = coded_data
        array = QueryTable[clusterIndex]
        # //find position
        pos = LSSSimplified.hashPos(key, clusterIndex) % (len(array))
        avgVal = array[pos]
        return avgVal