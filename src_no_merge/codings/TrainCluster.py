# -*- coding: utf-8 -*

import numpy as np
class TrainCluster (object):
	# /**
	#  * get positive
	#  * @param traces
	#  * @return
	#  */
    @staticmethod
    def getPositive(traces):
        positives = []
        for i in range(np.size(traces)):
            if traces[i]>0:
                positives.append(traces[i])
        return np.array(positives)
    # /**
    # * nonpositive
    # * @param traces
    # * @return
    # */
    @staticmethod
    def getNonPositive(traces):
        negatives = []
        for i in range(np.size(traces)):
            if traces[i] <= 0:
                negatives.append(traces[i])
        return np.array(negatives)