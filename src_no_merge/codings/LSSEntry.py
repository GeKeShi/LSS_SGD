# -*- coding: utf-8 -*


class LSSEntry(object):

	def __init__(self):
		self.counter = 0
		self.sum = 0
	
	# //public LSSEntry(int _c, double _s){
	# //	counter = _c;
	# //	sum = _s;
	# //}
	def getCounter(self):
		return self.counter

	def setSum(self, sum0):
		 self.sum = sum0

	def getSum(self):
		return self.sum

	# /**
	#  * assume unique id;
	#  * increase counter, sum
	#  * @param val
	#  */
	def update(self, val):
		# // TODO Auto-generated method stub
		self.counter += 1
		self.sum += val
        
	# /**
	#  * estimator
	#  * @return
	#  */
	def getAvgEstimator(self):
		# // TODO Auto-generated method stub
		if self.counter >= 1:
			return (self.sum)/(self.counter)
		else:
			return self.sum
