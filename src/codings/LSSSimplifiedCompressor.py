from LSSSimplified import *
# from VectorCompressor import *
import numpy as np
from EncodeChoicer import *
from HuffmanEncoder import *
import time

class LSSSimplifiedCompressor(object):
    
	clusterNumber = 50
	denseEncoders = HuffmanEncoder()
	# lssCKInstance = LSSSimplified()
	# N = 0
	# //Map<Key,Integer> shadowMapGlobal;
	# /**
	#  * direct mapping
	#  */
	# //boolean useDirectIndex = false;
	
	# //membership encoder
	# denseEncoders = BinaryEncoder()
	
	# //sparse key encode
	# private BinaryEncoder sparseKeyEncoder;
	# NumOfItems = 0
	
	# //cluster center
	# centers = []

	def __init__(self, _expectedNumItems, _clusterCount, _bucketCount, _clusterArrayChoiceMethod, traces):
		# /**
		#  * constructor of LSS sketch
		#  * @param _expectedNumItems
		#  * @param _clusterCount
		#  * @param _bucketCount
		#  * @param _expectedFP
		#  * @param traces
		#  * @param scaleFactor
		#  * @param bufferedWriter
		#  */
		# double meanVal = StatUtils.mean(traces);
		# //inverse of the mean as the acale factor
		#  //float scaleFactor = (float) (1.0/meanVal);
		# // double[] scaleFactor = {0,1};//batchNormalization(traces);
		 
		#  System.out.println("meanVal: "+meanVal+", samples: "+traces.length);
		# /**
		#  * init lss sketch
		#  */
		self.lssCKInstance = LSSSimplified(_expectedNumItems, _clusterCount, _bucketCount, _clusterArrayChoiceMethod)
		
		self.centers =  self.lssCKInstance.trainClusterAndSetArraySize(traces)
		# //normalized(traces,scaleFactor[0],scaleFactor[1]));
		
		# if encodeChoice == encodeChoicer.deltaAdaptive:	 
		# 	self.denseEncoders = DeltaAdaptiveEncoder()
		# #  //delta binary
		# elif encodeChoice == encodeChoicer.DeltaBinary:			 
		# 	self.denseEncoders= DeltaBinaryEncoder()	
		 
		# elif encodeChoice == encodeChoicer.Huffman:			
		# 	self.denseEncoders = HuffmanEncoder()
		self.encoded_index = None
		self.LSSTable_val = None
		self.NumOfItems = 0
		self.N = 0
		print("initialized LSS completed!")

	def compressDense(self, values):
		index = np.zeros(values.size, dtype=int)

		self.NumOfItems = values.size	
		# long t1 = System.currentTimeMillis();
		current_time = time.time()
		for i in range(values.size):
			
			keyVal = i+1#//key ？
				
			#	//Key key =new Key(ByteBuffer.allocate(4).putInt(val).array());
			 #	//mapped array
			#call new insert method
			pos = self.lssCKInstance.insert_cluster_lable(keyVal.to_bytes(4, byteorder='big'), values[i], self.centers)
			# pos = self.lssCKInstance.groupInputKV(values[i], self.centers)
			#	//System.out.println("$: "+i+", "+pos);
			#	//index
			index[i] = pos
		print('insert time:', time.time()-current_time)

		current_time = time.time()
		self.LSSTable_val = self.centers#use cluster centers to instead the lsstable
		print('get lsstable time:', time.time()-current_time)

		#//value index
		#long t2 = System.currentTimeMillis();
		current_time = time.time()
		self.encoded_index = LSSSimplifiedCompressor.denseEncoders.encode(index.tolist())
		print('encode index time:', time.time()-current_time)

		#long t3 = System.currentTimeMillis();
		
		#//index = null;
		#LOGGER.info("sketch: "+(t2-t1)+", keyEncode: "+(t3-t2));
	    #//System.out.println("Inserted: "+values.length);
		self.N = values.size

	def parallelcompressDense(self, values):
		self.compressDense(values)

	@staticmethod
	def decompressDense(code):
		if code['encode_flag']==True:
			vals = np.zeros(code['flatten_size'])
			# long t1 = System.currentTimeMillis();
			# //find keys in this array
			# int[] keysInGroup =this.denseEncoders.decode();
			keysInGroup = LSSSimplifiedCompressor.denseEncoders.decode(code['coded_index'])
			coded_data = code['coded_data']
			# hash_list = code['hash_list']
			# long t2 = System.currentTimeMillis();
			for keyIndex in range(len(keysInGroup)):
				keyVal = keyIndex+1
				arrayIndex = keysInGroup[keyIndex]
				# //Key key =new Key(ByteBuffer.allocate(4).putInt(keyVal).array());				
				
				# //index should be smaller by one
				#vals[keyVal - 1] = LSSSimplified.query_val(keyVal.to_bytes(4, byteorder='big'), arrayIndex, coded_data)
				vals[keyVal - 1] = coded_data[arrayIndex]
			# long t3 = System.currentTimeMillis();
			# LOGGER.info("decodeKey: "+(t2-t1)+", sketch: "+(t3-t2));
			vals = vals.reshape(code['shape'])
			keysInGroup = None
			return vals
		else:
			return code['coded_data']


	 
