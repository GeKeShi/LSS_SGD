extern "C"{
    // #include <stdio.h>
    // #define QUANTIZED_LEVEL 3
    #define WORKER_NUMBER 3
    /* BKDR Hash Function */
    __device__ unsigned int BKDR_hash(char *str){
        unsigned int seed = 131;
        unsigned int hash = 0;
        for(int i=0;i<4;i++){
            hash = hash * seed + (str[i]);
        }
        return (hash & 0x7FFFFFFF);
    }
    // change the type of keysInGroup_gpu, add quant_level,to add bit_unit
    __global__ void InsertKernel(float *lsstable_gpu, unsigned int *keysInGroup_gpu, float *grad_gpu, int *lsstable_size_gpu, float * cluster_center_gpu, int grad_num, int cluster_number, int quant_level, int max_table_size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int bit_uint = int(32/quant_level);// how many code in one longlong int, should be pass by the parameter, to do
         
        if (idx < grad_num){
            int col = idx % bit_uint;
            int row = int(idx / bit_uint);// code cordination
            float tmp = fabsf(cluster_center_gpu[0] - grad_gpu[idx]);
            int i;
            unsigned int tmp_index = 0;//closest cluster index
            for(i=1;i<cluster_number;i++){
                float tmp_diff = fabsf(cluster_center_gpu[i] - grad_gpu[idx]);
                if(tmp_diff < tmp){
                    tmp = tmp_diff;
                    tmp_index = i;
                }	    
            }
            
            unsigned int tmp_bits = tmp_index;
            tmp_bits <<= (quant_level * col);
            
            atomicOr(&(keysInGroup_gpu[row]), tmp_bits);
            // if(idx==15){
            //     printf("id %d, quant %d\n", idx,tmp_index);
            //     printf("id %d, tmp_bits %d\n", idx,tmp_bits);
            //     printf("id %d, keysInGroup_gpu %d\n", idx,keysInGroup_gpu[row]);
            // }
            // printf("id %d, quant %d\n", idx,keysInGroup_gpu[row]);
            
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
    /****
    __global__ void DecodeKernel(float *lsstable_gpu, unsigned char *keysInGroup_gpu, float *grad_gpu, int *lsstable_size_gpu, int grad_num, int cluster_number, int max_table_size){
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
    ****/
    
    // change the type of keysInGroup_gpu, add int quant_level, int workers_number, int keysInGroup_size, to add bit_unit

    __global__ void Decode_Multi_Kernel(float *lsstable_gpu, unsigned int *keysInGroup_gpu, float *grad_gpu, int *lsstable_size_gpu, int grad_num, int workers_number, int keysInGroup_size, int quant_level, uint cluster_number, int max_table_size){
        extern __shared__ float cache[]; //each row save the decode value from the concatenate keysInGroup and compute one grad, 128*WORKER_NUMBER
        
        int idx = threadIdx.x + blockIdx.x * blockDim.x; //to find the grad id
        int idy = threadIdx.y; //to find which worker's keysInGroup
        int bit_uint = int(32/quant_level);// how many code in one longlong int, should be pass by the parameter, to do
        if (idx < grad_num){
            int col = idx % bit_uint;
            int row = int(idx / bit_uint);// code cordination
            unsigned int tmp_bits = (cluster_number-1) << (quant_level * col);//mask
            tmp_bits &= keysInGroup_gpu[idy * keysInGroup_size + row];
            
            int group = tmp_bits >> (quant_level * col);//get the code
            // if(idx==(grad_num-1)){
            //     printf("id %d, pos %d\n", idx,idy * keysInGroup_size + row);
            //     printf("id %d, tmp_bits %d\n", idx,tmp_bits);
            //     printf("id %d, quant %d\n", idx,group);
            // }
            int hash_pos = (BKDR_hash((char *)&idx)) % (lsstable_size_gpu[group]);
            cache[threadIdx.x* workers_number + threadIdx.y] = lsstable_gpu[2*group*max_table_size + hash_pos];
        }
        __syncthreads();

        int i = workers_number/2;
        while(i != 0){
            if(threadIdx.y<i){
                cache[threadIdx.x* workers_number + threadIdx.y] += cache[threadIdx.x * workers_number + (threadIdx.y+i)];
            }
            __syncthreads();
            i /= 2;
        }

        if(idx < grad_num && threadIdx.y==0){
            grad_gpu[idx] = cache[threadIdx.x * workers_number + 0]/workers_number;
        }
    }
}