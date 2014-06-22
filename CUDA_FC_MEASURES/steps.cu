/*
 * steps.cpp
 *
 *  Description :   GPU steps 
 *
 *   Created on :   01.Mar.2012
 *       Author :   Orhan Firat
 *                  Department of Computer Engineering
 *                  Middle East Technical University
 *       E-mail :   orhan.firat@ceng.metu.edu.tr
 *
 *   Copyright, 2012, Orhan Firat
 *
 *      Vode An
 */



// Copyright (c) 2012 Orhan Firat
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "steps.h"

// 2D-Texture memory bindings for time-series
texture<TYPE, 2, cudaReadModeElementType> texSrc;
texture<TYPE, 2, cudaReadModeElementType> texDst;


// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const { 
            return x * x;
        }
};

/**
  * Utility function for writing matrix chunk to binary file
  *
  * @param out  			Output array of calculated correlation coefficients
  * @param sumX 			Sum of individual time series
  * @param sumXX 			Sum of squares of individual time series
  * @param tileSize 		Tile size for matrix chunk
  * @param tileIdx 			Horizontal index of current tile 
  * @param tileIdy			Vertical index of current tile
  * @param numTimesteps		Number of time series of a sample
  * @param numChunks		Total number of matrix chunks
  */	
__global__ void kernel_pearson_corr(TYPE* out, TYPE* sumX, TYPE* sumXX, 
									int tileSize, int tileIdx, int tileIdy, 
									int numTimesteps, int numChunks){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int linear_idx = idy*tileSize+ idx;

	int sumIdx = tileIdx*tileSize + idx;
	int sumIdy = tileIdy*tileSize + idy;

	if(idx<tileSize && idy<tileSize){

		TYPE accum = 0;
		for(int i=0; i<numTimesteps ; ++i){
			TYPE x = tex2D(texSrc,TYPE(i),TYPE(idx));
			TYPE y = tex2D(texDst,TYPE(i),TYPE(idy));
			accum += x*y;
		}
		__syncthreads();

		TYPE xbar = sumX[sumIdx]/numTimesteps;
		TYPE ybar = sumX[sumIdy]/numTimesteps;

		__syncthreads();

		TYPE xx = sumXX[sumIdx];
		TYPE yy = sumXX[sumIdy];

		out[linear_idx] = (accum - (numTimesteps*xbar*ybar))/
			(sqrtf( (xx - numTimesteps*xbar*xbar )*(yy - numTimesteps*ybar*ybar)));
	}
}

/**
  * Funtion to calculate sum of individual time series by a matrix-vector multiplication using cublas.
  *
  * @param meta		Metadata structure for hyper parameters and stuff		
  */
void step1_calculate_sumX(METADATA* meta){

    int MATRIX_CHUNKSIZE = meta->chunkSize;
    int NUM_SAMPLES      = meta->numSamples;
    int NUM_TIMESTEPS    = meta->numTimesteps;

	// calculate number of passes 
	int numPasses = (NUM_SAMPLES / MATRIX_CHUNKSIZE)+(NUM_SAMPLES % MATRIX_CHUNKSIZE == 0 ? 0 : 1);
			
	// allocate result vector on host
	meta->h_sumX = (TYPE*)malloc(NUM_SAMPLES*sizeof(TYPE));

	// allocate array and set to one for multiplication
	TYPE* ones_tmp = (TYPE*)malloc(NUM_SAMPLES*sizeof(TYPE));
	for(int i=0 ; i<NUM_SAMPLES ; ++i)
		ones_tmp[i]=1.0f;

	TYPE *d_ones_arr, *d_row_sum, *d_matrix_chunk, *h_matrix;
	h_matrix = meta->data;
	
	// allocate chunk matrix on device
	cudaMalloc((void**)&d_matrix_chunk, MATRIX_CHUNKSIZE*NUM_TIMESTEPS*sizeof(TYPE));

	// CUBLAS block
	{
		cublasInit();

		cublasAlloc( MATRIX_CHUNKSIZE, sizeof(TYPE), (void**) &d_ones_arr); //vector A
        cublasAlloc( MATRIX_CHUNKSIZE, sizeof(TYPE), (void**) &d_row_sum);  //vector B
        
		// transfer host data to device	for helper vector
		cublasSetVector( MATRIX_CHUNKSIZE, sizeof(TYPE), ones_tmp, 1, d_ones_arr, 1);
    
		start_timer();

		// process each chunk 
		for(int i=0 ; i<numPasses ; ++i){

			// get chunk indices
			int startInd = i*MATRIX_CHUNKSIZE;
			int endInd	 = startInd + ( (i+1==numPasses) ? (NUM_SAMPLES % MATRIX_CHUNKSIZE) : MATRIX_CHUNKSIZE ) - 1; 
			int numels	 = endInd - startInd + 1;

			// transfer host matrix chunk to device
			cudaMemcpy(d_matrix_chunk, &h_matrix[startInd*NUM_TIMESTEPS], numels*sizeof(TYPE)*NUM_TIMESTEPS, cudaMemcpyHostToDevice);
			cudaMemset(d_row_sum, 0, NUM_TIMESTEPS * sizeof(TYPE));

			// Perform matrix vector multiplication with cublas to obtain col sums
			cublasSgemv('T', NUM_TIMESTEPS, numels, 1, d_matrix_chunk, NUM_TIMESTEPS, d_ones_arr, 1, 0, d_row_sum, 1);
			
			// transfer device solution vector chunk to host
			cudaMemcpy(&meta->h_sumX[startInd], d_row_sum, numels*sizeof(TYPE), cudaMemcpyDeviceToHost);

		}

		stop_timer("STEP1::CUBLASSGEMV",1);

		cublasFree(d_ones_arr);
		cublasFree(d_row_sum);

		cublasShutdown();
	}
	cudaFree(d_matrix_chunk);
	free(ones_tmp);
}

/**
  *	Funtion to calculate sum of squares of individual time series by a matrix-vector 
  * multiplication using cublas. Square operation conducted using thrust library.
  * Written as a seperate function with almost the same code except thrust routines,
  * step1 and step2 are available for fully parallelisation for multiple GPUs 
  *	or the ones that can launch concurrent kernels.
  *
  * @param meta		Metadata structure for hyper parameters and stuff		
  */
void step2_calculate_sumXX(METADATA* meta){

    int MATRIX_CHUNKSIZE = meta->chunkSize;
    int NUM_SAMPLES      = meta->numSamples;
    int NUM_TIMESTEPS    = meta->numTimesteps;

	// calculate number of passes 
	int numPasses = (NUM_SAMPLES / MATRIX_CHUNKSIZE)+(NUM_SAMPLES % MATRIX_CHUNKSIZE == 0 ? 0 : 1);
			
	// allocate result vector on host
	meta->h_sumXX = (TYPE*)malloc( NUM_SAMPLES*sizeof(TYPE) );

	// allocate array and set to one for multiplication
	TYPE* ones_tmp = (TYPE*)malloc( NUM_SAMPLES*sizeof(TYPE) );
	for(int i=0 ; i<NUM_SAMPLES ; ++i)
		ones_tmp[i]=1.0f;

	TYPE *d_ones_arr, *d_row_sum, *d_matrix_chunk, *h_matrix;
	h_matrix = meta->data;

	// allocate chunk matrix on device
	cudaMalloc((void**)&d_matrix_chunk, MATRIX_CHUNKSIZE*NUM_TIMESTEPS*sizeof(TYPE));
	
	// CUBLAS block
	{
		cublasInit();

		cublasAlloc( MATRIX_CHUNKSIZE, sizeof(TYPE), (void**) &d_ones_arr); //vector A
        cublasAlloc( MATRIX_CHUNKSIZE, sizeof(TYPE), (void**) &d_row_sum);  //vector B
        
		// transfer host data to device	for helper vector
		cublasSetVector( MATRIX_CHUNKSIZE, sizeof(TYPE), ones_tmp, 1, d_ones_arr, 1);
    
		start_timer();

		// process each chunk 
		for(int i=0 ; i<numPasses ; ++i){

			// get chunk indices
			int startInd = i*MATRIX_CHUNKSIZE;
			int endInd	 = startInd + ( (i+1==numPasses) ? (NUM_SAMPLES % MATRIX_CHUNKSIZE) : MATRIX_CHUNKSIZE ) - 1; 
			int numels	 = endInd - startInd + 1;

			// transfer host matrix chunk to device
			cudaMemcpy(d_matrix_chunk, &h_matrix[startInd*NUM_TIMESTEPS], numels*sizeof(TYPE)*NUM_TIMESTEPS, cudaMemcpyHostToDevice);
			cudaMemset(d_row_sum, 0, NUM_TIMESTEPS * sizeof(TYPE));

			// square matrix chunk using thrust
			{
				thrust::device_ptr<TYPE> dev_ptr1(d_matrix_chunk);	

				square<TYPE> unary_op;
				thrust::transform(dev_ptr1, dev_ptr1+(numels*NUM_TIMESTEPS), dev_ptr1, unary_op);							
			}

			// Perform matrix vector multiplication with cublas to obtain col sums
			cublasSgemv('T', NUM_TIMESTEPS, numels, 1, d_matrix_chunk, NUM_TIMESTEPS, d_ones_arr, 1, 0, d_row_sum, 1);

			// transfer device solution vector chunk to host
			cudaMemcpy(&meta->h_sumXX[startInd], d_row_sum, numels*sizeof(TYPE), cudaMemcpyDeviceToHost);

		}

		stop_timer("STEP2::CUBLASSGEMV",2);


		cublasFree(d_ones_arr);
		cublasFree(d_row_sum);

		cublasShutdown();
	}
	cudaFree(d_matrix_chunk);
	free(ones_tmp);

}

/**
  * Major function to calculate pearson-correlation coefficient using previous
  * steps' results on device.
  *
  * @param meta		Metadata structure for hyper parameters and stuff		
  */
void step3_calculate_pearson_corr(METADATA* meta){

	clock_t start, stop, total=0;	// for writing output to disk

	PARAMETERS params;		

	calculate_parameters(&params, meta);

	int CHUNK_SIZE_IN_BYTES = params.tileSize*params.tileSize*sizeof(TYPE);

	// allocate resulting matrix chunk on device and host
	TYPE *d_matrix_chunk; 
	cudaMalloc((void**)&d_matrix_chunk, CHUNK_SIZE_IN_BYTES);
	TYPE *h_matrix_chunk = (TYPE*)malloc( CHUNK_SIZE_IN_BYTES );
	if(h_matrix_chunk==NULL){
		printf("I cannot allocate any more please stop!\n");
		exit(-1);
	}	

	// transfer sum(Xi) and sum(square(Xi)) arrays to device-memory
	// TODO try utilizing constant memory of texture memory 
	cudaMalloc((void**)&meta->d_sumX, sizeof(TYPE)*meta->numSamples);
	cudaMalloc((void**)&meta->d_sumXX, sizeof(TYPE)*meta->numSamples);
	Check_CUDA_Error("cudaMalloc");

	cudaMemcpy(meta->d_sumX, meta->h_sumX, sizeof(TYPE)*meta->numSamples, cudaMemcpyHostToDevice);
	cudaMemcpy(meta->d_sumXX, meta->h_sumXX, sizeof(TYPE)*meta->numSamples, cudaMemcpyHostToDevice);
	Check_CUDA_Error("cudaMemcpyHostToDevice");

	// allocate arrays for source and destination timeseries
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<TYPE>();
    cudaArray *cu_array_src,*cu_array_dst;		//TODO:these two can be merged
    cudaMallocArray( &cu_array_src, &channelDesc, meta->numTimesteps , params.tileSize); 
	cudaMallocArray( &cu_array_dst, &channelDesc, meta->numTimesteps , params.tileSize); 
	Check_CUDA_Error("cudaMallocArray");

	// set texture parameters
    texSrc.filterMode = cudaFilterModePoint;    texSrc.normalized = false;    
	texDst.filterMode = cudaFilterModePoint;    texDst.normalized = false;    

	// configure grids and blocks
	dim3 grid,block;
	block.x = BLOCK_X;
	block.y = BLOCK_Y;
	grid.x = (int)(ceil((double)params.tileSize / (double)block.x));
	grid.y = (int)(ceil((double)params.tileSize / (double)block.y));

	// start computing correaltion
	for(int i=0,ctr=0 ; i<params.numPass ; ++i ){

		// transfer source timeseries data and bind to texture 		
		CUDA_SAFE_CALL(cudaMemcpy2DToArray( cu_array_src, 0, 0, 
							&meta->data[i*params.tileSize*meta->numTimesteps], 
							sizeof(TYPE)*meta->numTimesteps, sizeof(TYPE)*meta->numTimesteps, 
							( i+1==params.numPass ? (meta->numSamples%params.tileSize) : params.tileSize),
							cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaBindTextureToArray( texSrc, cu_array_src, channelDesc));

		for(int j=0 ; j<i+1 ; ++j ,++ctr){

			start_timer();

			// transfer destination timeseries data and bind to texture
			CUDA_SAFE_CALL(cudaMemcpy2DToArray( cu_array_dst, 0, 0, 
							&meta->data[j*params.tileSize*meta->numTimesteps], 
							sizeof(TYPE)*meta->numTimesteps, sizeof(TYPE)*meta->numTimesteps, 
							((i+1==params.numPass && j+1==params.numPass) ? (meta->numSamples%params.tileSize) : params.tileSize),
							cudaMemcpyHostToDevice));

			CUDA_SAFE_CALL(cudaBindTextureToArray( texDst, cu_array_dst, channelDesc));
			CUDA_SAFE_CALL(cudaMemset((void*)d_matrix_chunk, 0x0 , CHUNK_SIZE_IN_BYTES));
			memset(h_matrix_chunk, 0x0 , CHUNK_SIZE_IN_BYTES);

			/***********************
			 * start doing the job *
			 **********************/	

			kernel_pearson_corr<<<grid,block>>>(d_matrix_chunk, meta->d_sumX, meta->d_sumXX, params.tileSize, i, j, meta->numTimesteps, params.numPass);
			Check_CUDA_Error("kernel_pearson_corr");

			/***********************
			 * stop doing the job  *
			 **********************/	

			// copy result from device to host 		
			CUDA_SAFE_CALL(cudaMemcpy(h_matrix_chunk, d_matrix_chunk, CHUNK_SIZE_IN_BYTES , cudaMemcpyDeviceToHost));			

			// unbind current texture 
			// TODO : try not unbinding
			CUDA_SAFE_CALL(cudaUnbindTexture(texDst));					

			stop_timer("STEP3::CORR",3);

			printf("numPass:%d curr:[%d][%d] ctr:%d\n",params.numPass,i,j,ctr);		
			fflush(stdout);

			start = clock();

			if(meta->isOutputBinary)
				write_matChunk_to_binFile(h_matrix_chunk, params.tileSize, params.tileSize,i,j, meta->outputDir);
			else // ascii-txt
				write_matChunk_to_file(h_matrix_chunk, params.tileSize, params.tileSize,i,j, meta->outputDir);

			stop = clock();
		
			total += (float)(stop-start);
			
		}

		// unbind current texture	
		// TODO : try not unbinding
		CUDA_SAFE_CALL(cudaUnbindTexture(texSrc));
	}

	set_host_timer("STEP4::WRITE_DATA",(float)(total),2);

	// clean up
	CUDA_SAFE_CALL(cudaFreeArray(cu_array_src));
	CUDA_SAFE_CALL(cudaFreeArray(cu_array_dst));
	CUDA_SAFE_CALL(cudaFree(d_matrix_chunk));
	free(h_matrix_chunk);
}
