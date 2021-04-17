# -*- coding: utf-8 -*
from LSSSimplified import *
# from VectorCompressor import *
import numpy as np
from EncodeChoicer import *
from HuffmanEncoder import *
import time, struct
import sys
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
from Bitmap import BitMap
import os
import math
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
		# print("{} is the cluster center".format(self.centers))
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
		# self.key_bit = BitMap()
		# self.N = 0
		# print("initialized LSS completed!")

	def compressDense(self, values):
		cluster_number =self.centers.size
		NumOfItems = values.size
		# index = np.zeros((NumOfItems, cluster_number), dtype=int)

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

		# keep the tmp_index be the index of most close cluster center in (keys_group[idx][]), tmp is the closest error, keys_group_gpu[idx*cluster_number + i]=1 if it is the tmp_index, =0 if not.
		kernel_code = """
		/* BKDR Hash Function */
		__device__ unsigned int BKDR_hash(char *str)
		{
			unsigned int seed = 131;
			unsigned int hash = 0;
			for(int i=0;i<4;i++)
			{
				hash = hash * seed + (str[i]);
			}
			return (hash & 0x7FFFFFFF);
		}

		__global__ void InsertKernel(float *lsstable_gpu, float *keys_group_gpu, float *grad_gpu, int *lsstable_size_gpu, int grad_num, int cluster_number, int max_table_size)
		{
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx < grad_num){
				float tmp = fabsf(keys_group_gpu[idx*cluster_number + 0] - grad_gpu[idx]);
				int i;
				int tmp_index = 0;
				keys_group_gpu[idx*cluster_number + 0] = 1;
				for(i=1;i<cluster_number;i++){
					if(fabsf(keys_group_gpu[idx*cluster_number + i] - grad_gpu[idx]) < tmp){
						tmp = fabsf(keys_group_gpu[idx*cluster_number + i] - grad_gpu[idx]);
						keys_group_gpu[idx*cluster_number + i] = 1;
						keys_group_gpu[idx*cluster_number + tmp_index] = 0;
						tmp_index = i;
					}	
					else{
						keys_group_gpu[idx*cluster_number + i] = 0;
					}
				}
				int hash_pos = (BKDR_hash((char *)&idx)) % (lsstable_size_gpu[tmp_index]);
				atomicAdd(&(lsstable_gpu[tmp_index*2*max_table_size + hash_pos]), grad_gpu[idx]);
				atomicAdd(&(lsstable_gpu[(tmp_index*2+1)*max_table_size + hash_pos]), 1.0);

			}
		}

		__global__ void get_lsstable_kernel(float *lsstable_gpu, int *lsstable_size_gpu, int cluster_number, int max_table_size){
			int idx = blockDim.x * blockIdx.x + threadIdx.x;
			int idy = blockIdx.y;
			if(idx < lsstable_size_gpu[idy]){
				if(lsstable_gpu[(idy*2+1)*max_table_size + idx]!=0){
					lsstable_gpu[(idy*2)*max_table_size + idx] /= lsstable_gpu[(idy*2+1)*max_table_size + idx];
				}
			}
		}		
		"""
		start = driver.Event()
		end = driver.Event()
		start.record()
		lsstable_size = np.array([len(self.lssCKInstance.LSSTable[i]) for i in range(cluster_number)], dtype=np.int32)
		max_table_size = np.amax(lsstable_size)
		# print("lsstable size {}, maxtable size{}".format(lsstable_size, max_table_size))
		# lsstable_size_gpu = gpuarray.to_gpu(lsstable_size)
		# self.LSSTable_val = np.zeros((2*cluster_number, max_table_size), dtype=np.float32)
		lsstable_cpu = np.zeros((2*cluster_number, max_table_size), dtype=np.float32)
		quant_level = int(math.ceil((math.log(cluster_number,2))))
		bit_unit = 32//quant_level #how many code in each float32
		keys_group = np.zeros((NumOfItems+bit_unit-1)//bit_unit, dtype=np.uint32)
		cluster_center_cpu = self.centers.reshape(1,-1).astype(np.float32)
		# print("quant level {}, bit_unit {}, keys {}, cluster_center {}".format(quant_level, bit_unit, keys_group[:10],cluster_center_cpu))
		# lsstable_gpu = gpuarray.zeros((2*cluster_number, max_table_size), dtype=np.float32)
		# keys_group = np.repeat(self.centers.reshape(1,-1).astype(np.float32), values.size, axis=0)
		# print("keys_group repeat center {}".format(keys_group[:10]))
		# keys_group_gpu = gpuarray.to_gpu(keys_group).astype(np.float32)
		grad_cpu = values.astype(np.float32)
		# grad_gpu = gpuarray.to_gpu(values).astype(np.float32)

		# JIT compile
		# mod = compiler.SourceModule(kernel_code, keep=True, cache_dir='../')

		# pre-compile
		mod = driver.module_from_file(os.path.join('/THL5/home/daodao/gks/Sketch_DNN/Collect_Gradients/ATOMO/src_no_merge/codings', 'LSSkernel.cubin'))

		insertkernel = mod.get_function("InsertKernel")
		blocksize = (1024,1,1)
		grid_size = ((NumOfItems+1023)//1024, 1)
		#__global__ void InsertKernel(float *lsstable_gpu, uint32_t *keysInGroup_gpu, float *grad_gpu, int *lsstable_size_gpu, float * cluster_center_gpu, int grad_num, int cluster_number, int quant_level, int max_table_size)
		insertkernel(driver.InOut(lsstable_cpu), driver.InOut(keys_group), driver.In(grad_cpu), driver.In(lsstable_size), driver.In(cluster_center_cpu), np.int32(NumOfItems), np.int32(cluster_number), np.int32(quant_level), np.int32(max_table_size), block=blocksize, grid=grid_size)
		# keys_group_gpu.get(ary=keys_group)
		end.record()
		end.synchronize()
		secs = start.time_till(end)*1e-3
		# print ("SourceModule assign cluster time :")
		# print ("%fs " % (secs))
		# print("keys group before encode {}".format(keys_group[:100]))
		# keys_group = keys_group.astype(np.uint8)
		# self.key_bit.pack(keys_group)
		# print("keys group after encode {}, size of code {}".format(keys_group[:100], sys.getsizeof(keys_group)))



		blocksize = (128,1,1)
		grid_size = ((max_table_size+127)//128, cluster_number)

		getlsstable_kernel = mod.get_function("get_lsstable_kernel")
		
		start.record()
		getlsstable_kernel(driver.InOut(lsstable_cpu), driver.In(lsstable_size), np.int32(cluster_number), np.int32(max_table_size), block=blocksize, grid=grid_size)
		# lsstable_gpu.get(ary=self.LSSTable_val)
		self.LSSTable_val = lsstable_cpu
		end.record()
		end.synchronize()
		secs = start.time_till(end)*1e-3
		# print ("SourceModule gettable time :")
		# print ("%fs " % (secs))
		# print("lsstable {}, size of lsstable {}".format(self.LSSTable_val, self.LSSTable_val.shape))
		# self.LSSTable_val = self.LSSTable_val[::2]#get the even lines
		# print("lsstable {}, size of lsstable {}".format(self.LSSTable_val, sys.getsizeof(self.LSSTable_val)))
		# print('insert time:', time.time()-current_time)
		code = {'coded_data':self.LSSTable_val, 'coded_index':keys_group, 'lsstable_size': lsstable_size, 'flatten_size':NumOfItems}

		# lsstable_gpu.gpudata.free()
		# lsstable_size_gpu.gpudata.free()
		# keys_group_gpu.gpudata.free()
		# grad_gpu.gpudata.free()
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
		keysInGroup: containing the cluster index of each grad, numpy shape(grad_number, cluster_number), np.uint8
		grad_number : size of gradients 
		lsstable_size: save the len of each hash table, numpy shape(cluster_number,) np.int
		'''
		# if code['encode_flag']==True:

		kernel_code_template = """
		#define TOTAL_CLUSTER_NUMBER 8
		/* BKDR Hash Function */
		__device__ unsigned int BKDR_hash(char *str)
		{
			unsigned int seed = 131;
			unsigned int hash = 0;
			for(int i=0;i<4;i++)
			{
				hash = hash * seed + (str[i]);
			}
			return (hash & 0x7FFFFFFF);
		}

		__global__ void DecodeKernel(float *lsstable_gpu, unsigned char *keysInGroup_gpu, float *grad_gpu, int *lsstable_size_gpu, int grad_num, int cluster_number, int max_table_size)
		{
			__shared__ float cache[128][TOTAL_CLUSTER_NUMBER];
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			int idy = threadIdx.y;

			if (idx < grad_num){
				int group_times = keysInGroup_gpu[idx*cluster_number + idy];
				int hash_pos = (BKDR_hash((char *)&idx)) % (lsstable_size_gpu[idy]);
				cache[threadIdx.x][threadIdx.y] = lsstable_gpu[2*idy*max_table_size + hash_pos] * group_times;
			}
			__syncthreads();

			int i = TOTAL_CLUSTER_NUMBER/2;
			while(i != 0){
				if(threadIdx.y<i){
					cache[threadIdx.x][threadIdx.y] += cache[threadIdx.x][threadIdx.y+i];
				}
				__syncthreads();
				i /= 2;
			}

			if(idx < grad_num && threadIdx.y==0){
				grad_gpu[idx] = cache[threadIdx.x][0]/TOTAL_CLUSTER_NUMBER;
			}
		}		
		"""
		start = driver.Event()
		end = driver.Event()
		start.record()
		grad_cpu = np.zeros((grad_number,), dtype=np.float32)
		cluster_number = len(lsstable_size)
		quant_level = int(math.ceil((math.log(cluster_number,2))))
		bit_unit = 32//quant_level
		keysInGroup_size = (grad_number+bit_unit-1)//bit_unit #one keys_group szie
		if keysInGroup.ndim == 1:
			workers_number = 1
		elif keysInGroup.ndim == 2:
			workers_number = keysInGroup.shape[0]
		# print('worker number {},keysInGroup_size {},type{}'.format(workers_number,keysInGroup_size, keysInGroup.dtype))
		# long t1 = System.currentTimeMillis();
		# //find keys in this array
		# int[] keysInGroup =this.denseEncoders.decode();
		# keysInGroup = LSSSimplifiedCompressor.denseEncoders.decode(code['coded_index'], code['codebook'])
		# coded_data = lsstable
		# hash_list = code['hash_list']
		# long t2 = System.currentTimeMillis();

		# lsstable_gpu = gpuarray.to_gpu(lsstable).astype(np.float32)
		# lsstable_size_gpu = gpuarray.to_gpu(lsstable_size).astype(np.float32)
		max_table_size = np.amax(lsstable_size)
		# grad_gpu = gpuarray.zeros((grad_number,), dtype=np.float32)
		# keysInGroup_gpu = gpuarray.to_gpu(keysInGroup).astype(np.uint8)

		# JIT compile
		# kernel_code = kernel_code_template % (cluster_number, cluster_number, cluster_number)

		# mod = compiler.SourceModule(kernel_code_template, keep=True)

		# Pre-compile
		mod = driver.module_from_file(os.path.join('/THL5/home/daodao/gks/Sketch_DNN/Collect_Gradients/ATOMO/src_allgrad_encode/codings', 'LSSkernel.cubin'))


		decodekernel = mod.get_function("Decode_Multi_Kernel")
		blocksize = (64,workers_number,1)
		grid_size = ((grad_number+127)//128, 1)

		# keysInGroup_gpu = driver.mem_alloc(keysInGroup.nbytes)
		# driver.memcpy_htod(keysInGroup_gpu, keysInGroup)
		# __global__ void Decode_Multi_Kernel(float *lsstable_gpu, uint32_t *keysInGroup_gpu, float *grad_gpu, int *lsstable_size_gpu, int grad_num, int workers_number, int keysInGroup_size, int quant_level, uint cluster_number, int max_table_size)
		decodekernel(driver.In(lsstable), driver.In(keysInGroup), driver.InOut(grad_cpu), driver.In(lsstable_size), np.int32(grad_number), np.int32(workers_number), np.int32(keysInGroup_size), np.int32(quant_level), np.uint32(cluster_number), np.int32(max_table_size), block=blocksize, grid=grid_size, shared=4*128*workers_number)
		# grad_gpu.get(ary=grad_cpu)#transfer the data to cpu from GPU
		end.record()
		end.synchronize()
		secs = start.time_till(end)*1e-3
		# print ("SourceModule decode time :")
		# print ("%fs " % (secs))
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
		# lsstable_gpu.gpudata.free()
		# lsstable_size_gpu.gpudata.free()
		# keysInGroup_gpu.gpudata.free()
		# grad_gpu.gpudata.free()
		return grad_cpu
		# else:
			# return code['coded_data']


	 
