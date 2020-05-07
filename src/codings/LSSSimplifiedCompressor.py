# -*- coding: utf-8 -*
from LSSSimplified import *
# from VectorCompressor import *
import numpy as np
from EncodeChoicer import *
from HuffmanEncoder import *
import time

from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
from Bitmap import BitMap
class LSSSimplifiedCompressor(object):
    
	# clusterNumber = 50
	# denseEncoders = HuffmanEncoder()
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
		# self.encoded_index = None
		# self.codebook = None
		self.LSSTable_val = None
		# self.NumOfItems = 0
		self.key_bit = BitMap()
		# self.N = 0
		print("initialized LSS completed!")

	def compressDense(self, values):
		cluster_number =self.centers.size
		NumOfItems = values.size	
		index = np.zeros((NumOfItems, cluster_number), dtype=int)

		# long t1 = System.currentTimeMillis();
		current_time = time.time()
		# for i in range(values.size):
			
		# 	keyVal = i+1#//key ？
				
		# 	#	//Key key =new Key(ByteBuffer.allocate(4).putInt(val).array());
		# 	 #	//mapped array
			
		# 	pos = self.lssCKInstance.insert(keyVal.to_bytes(4, byteorder='big'), values[i], self.centers)
		# 	# pos = self.lssCKInstance.groupInputKV(values[i], self.centers)
		# 	#	//System.out.println("$: "+i+", "+pos);
		# 	#	//index
		# 	index[i] = pos
		kernel_code = """
		__global__ void InsertKernel(float *lsstable_gpu, int *keys_group_gpu, float *grad_gpu, int *lsstable_size_gpu, int grad_num, int cluster_number)
		{
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			int i;
			if (idx < grad_num){
				tmp = fabsf(keys_group_gpu[idx][0] - grad_gpu[idx]);
				tmp_index = 0;
				for(i=1;i++;i<cluster_number){
					if(fabsf(keys_group_gpu[idx][i] - grad_gpu[idx]) < tmp){
						tmp = fabsf(keys_group_gpu[idx][i] - grad_gpu[idx]);
						keys_group_gpu[idx][i] = 1;
						keys_group_gpu[idx][tmp_index] = 0;
						tmp_index = i;
					}	
					else{
						keys_group_gpu[idx][i] = 0;
					}
				}
				hash_pos = (idx % 2013) % (lsstable_size_gpu[tmp_index]);
				atomicAdd(&(lsstable_gpu[tmp_index*2][hash_pos]), grad_gpu[idx]);
				atomicAdd(&(lsstable_gpu[tmp_index*2+1][hash_pos]), 1.0);

			}
		}

		__globals__ void get_lsstable_kernel(float *lsstable_gpu, int *lsstable_size_gpu){
			int idx	= blockDim.x * blockIdx.x + threadIdx.x;
			int idy = blockIdx.y;
			if(idx < lsstable_size_gpu[idy]){
				if(lsstable_gpu[idy*2+1][idx]!=0){
					lsstable_gpu[idy*2][idx] /= lsstable_gpu[idy*2+1][idx];
				}
			}
		}		
		"""
		start = driver.Event()
		end = driver.Event()
		start.record()
		lsstable_size = np.array([len(self.lssCKInstance.LSSTable[i]) for i in range(cluster_number)], dtype=np.int)
		max_table_size = np.amax(lsstable_size)
		lsstable_size_gpu = gpuarray.to_gpu(lsstable_size)
		self.LSSTable_val = np.zeros((2*cluster_number, max_table_size), dtype=np.float32)
		lsstable_gpu = gpuarray.zeros((2*cluster_number, max_table_size), dtype=np.float32)
		keys_group = np.repeat(self.centers.reshape(1,-1).astype(np.float32), values.size, axis=0)
		keys_group_gpu = gpuarray.to_gpu(keys_group, dtype=np.float32)
		grad_gpu = gpuarray.to_gpu(values).astype(np.float32)
		mod = compiler.SourceModule(kernel_code)
		insertkernel = mod.get_function("InsertKernel")
		blocksize = (1024,1,1)
		grid_size = ((NumOfItems+1023)//1024, 1)
		insertkernel(lsstable_gpu, keys_group_gpu, grad_gpu, lsstable_size_gpu, NumOfItems, cluster_number, block=blocksize, grid=grid_size)
		keys_group_gpu.get(ary=keys_group)
		end.record()
		end.synchronize()
		secs = start.time_till(end)*1e-3
		print ("SourceModule insert time :")
		print ("%fs " % (secs))
		keys_group = keys_group.astype(np.uint8)
		self.key_bit.pack(keys_group)



		blocksize = (128,1,1)
		grid_size = ((max_table_size+127)//128, cluster_number)
		getlsstable_kernel = mod.get_function("get_lsstable_kernel")
		start.record()
		getlsstable_kernel(lsstable_gpu, lsstable_size_gpu, block=blocksize, grid=grid_size)
		lsstable_gpu.get(ary=self.LSSTable_val)
		end.record()
		end.synchronize()
		secs = start.time_till(end)*1e-3
		print ("SourceModule insert time :")
		print ("%fs " % (secs))
		self.LSSTable_val = self.LSSTable_val[::2]

		print('insert time:', time.time()-current_time)
		code = {'coded_data':self.LSSTable_val, 'coded_index':self.key_bit, 'lsstable_size': lsstable_size, 'flatten_size':NumOfItems}
		return code
		# current_time = time.time()
		# # self.LSSTable_val = self.lssCKInstance.get_lsstable()
		# print('get lsstable time:', time.time()-current_time)

		# #//value index
		# #long t2 = System.currentTimeMillis();
		# current_time = time.time()
		# # self.codebook, self.encoded_index = LSSSimplifiedCompressor.denseEncoders.encode(index.tolist())
		# print('encode index time:', time.time()-current_time)

		#long t3 = System.currentTimeMillis();
		
		#//index = null;
		#LOGGER.info("sketch: "+(t2-t1)+", keyEncode: "+(t3-t2));
	    #//System.out.println("Inserted: "+values.length);
		# self.N = values.size

	# def parallelcompressDense(self, values):
	# 	self.compressDense(values)

	@staticmethod
	def decompressDense(lsstable, keysInGroup, grad_number, lsstable_size):
		'''
		param 
		lsstable:  several hash table saving each average grad, numpy shape(cluster_number, max(lsstable_size)), np.float32
		keysInGroup: containing the cluster index of each grad, numpy shape(gra_number, cluster_number), np.uint8
		grad_number : size of gradients 
		lsstable_size: save the len of each hash table, numpy shape(cluster_number,) np.int
		'''
		# if code['encode_flag']==True:

		kernel_code_template = """
		__global__ void DecodeKernel(float *lsstable_gpu, int *keysInGroup_gpu, float *grad_gpu, int *lsstable_size_gpu, int grad_num)
		{
			__shared__ float cache[128][%(CLUSTER_NUM)s]
			int idx = threadIdx.x + blockIdx.x * BlockDim.x;
			int idy = threadIdx.y;

			if (idx < grad_num){
				group_times = keysInGroup_gpu[idx][idy];
				hash_pos = (idx % 2013) % (lsstable_size_gpu[idy]);
				cache[threadIdx.x][threadIdx.y] = lsstable_gpu[idy][hash_pos] * group_times;
			}
			__syncthreads();

			int i = %(CLUSTER_NUM)s/2;
			while(i != 0){
				if(threadIdx.y<i){
					cache[threadIdx.x][threadIdx.y] += cache[threadIdx.x][threadIdx.y+i];
				}
				__syncthreads();
				i /= 2;
			}

			if(idx < grad_num && threadIdx.y==0){
				grad_gpu[idx] = cache[threadIdx.x][0]/%(CLUSTER_NUM)s
			}
		}		
		"""
		start = driver.Event()
		end = driver.Event()
		start.record()
		grad_cpu = np.zeros(grad_number, dtype=np.float32)
		cluster_number = len(lsstable_size)
		# long t1 = System.currentTimeMillis();
		# //find keys in this array
		# int[] keysInGroup =this.denseEncoders.decode();
		# keysInGroup = LSSSimplifiedCompressor.denseEncoders.decode(code['coded_index'], code['codebook'])
		# coded_data = lsstable
		# hash_list = code['hash_list']
		# long t2 = System.currentTimeMillis();
		lsstable_gpu = gpuarray.to_gpu(lsstable, dtype=np.float32)
		lsstable_size_gpu = gpuarray.to_gpu(lsstable, dtype=np.float32)
		grad_gpu = gpuarray.zeros((grad_number,), dtype=np.float32)
		keysInGroup_gpu = gpuarray.to_gpu(keysInGroup).astype(np.float32)
		kernel_code = kernel_code_template % {
			'CLUSTER_NUM': cluster_number
		}
		mod = compiler.SourceModule(kernel_code)
		insertkernel = mod.get_function("DecodeKernel")
		blocksize = (128,cluster_number,1)
		grid_size = ((grad_number+127)//128, 1)
		decodekernel(lsstable_gpu, keysInGroup_gpu, grad_gpu, lsstable_size_gpu, grad_number, block=blocksize, grid=grid_size)
		grad_gpu.get(ary=grad_cpu)
		end.record()
		end.synchronize()
		secs = start.time_till(end)*1e-3
		print ("SourceModule insert time :")
		print ("%fs " % (secs))
		# for keyIndex in range(len(keysInGroup)):
		# 	keyVal = keyIndex+1
		# 	arrayIndex = keysInGroup[keyIndex]
		# 	# //Key key =new Key(ByteBuffer.allocate(4).putInt(keyVal).array());				
			
		# 	# //index should be smaller by one
		# 	vals[keyVal - 1] = LSSSimplified.query_val(keyVal.to_bytes(4, byteorder='big'), arrayIndex, coded_data)
		# # long t3 = System.currentTimeMillis();
		# # LOGGER.info("decodeKey: "+(t2-t1)+", sketch: "+(t3-t2));
		# # vals = vals.reshape(code['shape'])
		# keysInGroup = None
		return grad_cpu
		# else:
			# return code['coded_data']


	 
