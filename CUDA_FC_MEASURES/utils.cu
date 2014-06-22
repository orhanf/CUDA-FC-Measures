/*
 * utils.cpp
 *
 *  Description :   Utility functions for the program 
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

#include <getopt.h>
#include "utils.h"

struct timer tm;

/**
  * Reads matrix from file with fiven filename
  *
  * @param filename	Filename of matrix 		
  * @param row		Rows of matrix
  * @param col		Columns of matrix	
  */
TYPE* read_mat_from_file(const char* filename,int rows, int cols){
		
	FILE* ftp=0;
	
	while(1){
		ftp=fopen(filename,"r+");
		if(ftp)
			break;
	}

	TYPE* f = (TYPE*)malloc(sizeof(TYPE)*rows*cols); 

	for(int i=0; i<rows ; ++i)
		for(int j=0; j<cols ;++j)
			fscanf_s(ftp, "%f", &f[i*cols+j]);

	fclose(ftp);
	
	return f;
}

/**
  * Prints given matrix
  *
  * @param mat	Matrix 		
  * @param row	Rows of matrix
  * @param col	Columns of matrix	
  * @param lda	Leading dimension of matrix	
  */
void print_matrix(TYPE* mat,int row,int col, int lda)
{
		for(int i=0 ; i<row ; ++i)
		{		
			for(int j=0 ; j<col ; ++j)
				printf("%.4f\t",mat[i*lda+j]);
			printf("\n");
		}
}

/**
  * Usage
  *
  * @param 			
  */
void usage(void) {

    printf("Usage: \n");		
	printf(" CUDA_FC.exe -n numSamples -t timeLength -i inputFile [switches]\n");		
	printf("   -i inputFile  : input file name for data. \n" );
	printf("                   Data matrix must be in the form n x t, where  \n");
	printf("                   n is the number of samples (rows)\n");
	printf("                   t is the length of timeseries (columns)\n");
	printf("   -n numSamples : total number of samples in the experiment\n");
	printf("   -t timeLength : length of timeseries of each sample\n");
	printf("   -c chunkSize  : chunk size of grid used in CUBLAS also\n");
    printf("                   matrix tile size of grid used in CUDA (dafeault 4096)\n");
    printf("   -o foldername : folder name for output matrices (default current dir)\n" );
    printf("   -b isBinary   : output file format, binary(1) or ascii-txt(0) (default 1)\n"); 
    printf("   -d            : enable debug mode (default no)\n" );
	printf("   -h            : help :)\n" );    
	fflush(stdout);
}

/**
  * Usage
  *
  * @param 			
  */
void help(void){
	usage();
	exit(-1);
}

/**
  * Init for main
  *
  * @param argc		
  * @param argv		
  */
METADATA* init(int argc, char** argv){

	// variables
    extern char*   optarg;
    extern int     optind;
    int     opt;

	METADATA* tmpMeta = (METADATA*)malloc(sizeof(METADATA));

	// default parameters
	tmpMeta->_debug = 0;
	tmpMeta->chunkSize = DEFAULT_MATRIX_CHUNKSIZE;
	tmpMeta->tileSize = DEFAULT_MATRIX_CHUNKSIZE;
	tmpMeta->outputDir = (char*)malloc(256*sizeof(char));
	tmpMeta->outputDir = "./"; 
	tmpMeta->inputFile = (char*)malloc(256*sizeof(char));
	tmpMeta->isOutputBinary = 1;

    // process arguments
    while ( (opt=getopt(argc,argv,"n:t:c:i:o:b:dh?"))!= EOF) {
        switch (opt) {
            case 'n': tmpMeta->numSamples=atoi(optarg);
                      break;
			case 't': tmpMeta->numTimesteps=atoi(optarg);
                      break;
			case 'c': tmpMeta->chunkSize=atoi(optarg);
					  tmpMeta->tileSize = tmpMeta->chunkSize;	
                      break;                        
            case 'i': tmpMeta->inputFile=optarg;
                      break;
            case 'o': tmpMeta->outputDir=optarg;
                      break;
            case 'd': tmpMeta->_debug = 1;
                      break;
			case 'b': atoi(optarg)==0 ? tmpMeta->isOutputBinary = 0 : tmpMeta->isOutputBinary = 1;
                      break;
			case '?': help();
                      break;
			case 'h': help();
                      break;
            default:  help();
                      break;
        }
    }
   
    // display informations
    if(tmpMeta->_debug){

		printf("\n\n/***********************************************************/");
		printf(  "\n/*********************USING PARAMETERS**********************/");
		printf(  "\n/***********************************************************/\n");

        printf("Number of samples        : %d\n", tmpMeta->numSamples );
		printf("Number of timesteps      : %d\n", tmpMeta->numTimesteps);
        printf("Chunk size for in CUBLAS : %d\n", tmpMeta->chunkSize );               
		printf("Tile size used in CUDA   : %d\n", tmpMeta->tileSize );               

		printf("Precision                : %s\n", (sizeof(TYPE)==sizeof(float) ? "single" : "double"));

		printf("Input file               : %s\n", tmpMeta->inputFile );
        printf("Output directory         : %s\n", tmpMeta->outputDir );
        printf("Output format            : %s\n", tmpMeta->isOutputBinary ? "binary" : "ascii-txt");

		printf(  "/***********************************************************/\n");
		fflush(stdout);
    }

	clock_t start, stop;

	start = clock();
		
	tmpMeta->data = read_mat_from_file(tmpMeta->inputFile, tmpMeta->numSamples, tmpMeta->numTimesteps);

	stop = clock();
	set_host_timer("STEP0::READ_DATA",(float)(stop-start),1);

	return tmpMeta;
}

/**
  * Starts timer for CUDA events.
  *
  */
void start_timer(void){
	tm.et=0;
	cudaEventCreate(&tm.start);    
	cudaEventCreate(&tm.stop);
	cudaEventRecord(tm.start, 0);	
}

/**
  * Sets host timer for CPU events.
  *
  * @param msg		Message for log
  * @param time		Elapsed time
  */
void set_host_timer(char* msg,float time, int step){
	
	switch(step){
		case 1:
			tm.times_host_read[msg] = time;
			break;
		case 2:
			tm.times_host_write[msg] = time;
			break;	
	}

	
}

/**
  * Stops timer with given time step and message for CUDA events, logging purposes.
  *
  * @param msg		Message for log
  * @param step		Current step for algorithm	
  */
void stop_timer(char* msg,int step){

	cudaEventRecord(tm.stop, 0);		
	cudaEventSynchronize(tm.stop);		
	cudaEventElapsedTime(&tm.et, tm.start, tm.stop);
	cudaEventDestroy(tm.start);     cudaEventDestroy(tm.stop);
	
	switch(step){
		case 1:
			tm.times1[msg] = tm.et;		  
			break;
		case 2:
			tm.times2[msg] = tm.et;		  
			break;
		case 3:
			tm.times3[msg] = tm.et;		  
			break;
	}
		
}

/**
  * Print time statistics to the given stream.
  *
  * @param stream	Output stream
  */
void print_timer(FILE *stream){

    // printg device props
    cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp,  0); 
	printDevProp(stream, deviceProp);

    
    // print time measurements
	std::map<char*, float>::iterator iter;
	
	float accum = 0;
	float overall = 0;

	fprintf_s(stream,"\n\n/***********************************************************/");
	fprintf_s(stream,  "\n/*********************TIME MEASUREMENTS*********************/");
	fprintf_s(stream,  "\n/***********************************************************/\n");

	fprintf_s(stream,  "\n/************************ON DEVICE**************************/");

	fprintf_s(stream,"\nSTEP 1\n");	
	for( iter = tm.times1.begin(); iter!=tm.times1.end(); ++iter){
		fprintf_s(stream, "Elapsed time for %s : [%f]ms\n",iter->first,iter->second); 
		accum += iter->second;
	}
	fprintf_s(stream,"TOTAL ELAPSED TIME: [%f]\n",accum);	
	overall += accum; accum = 0;

	fprintf_s(stream,"\nSTEP 2\n");
	for( iter = tm.times2.begin(); iter!=tm.times2.end(); ++iter){
		fprintf_s(stream, "Elapsed time for %s : [%f]ms\n",iter->first,iter->second); 
		accum += iter->second;
	}
	fprintf_s(stream,"TOTAL ELAPSED TIME: [%f]\n",accum);	
	overall += accum; accum = 0;

	fprintf_s(stream,"\nSTEP 3\n");
	for( iter = tm.times3.begin(); iter!=tm.times3.end(); ++iter){
		fprintf_s(stream, "Elapsed time for %s : [%f]ms\n",iter->first,iter->second); 
		accum += iter->second;
	}
	fprintf_s(stream,"TOTAL ELAPSED TIME: [%f]\n",accum);	
	overall += accum; accum = 0;
	
	fprintf_s(stream,  "\n/*************************ON HOST***************************/");

	fprintf_s(stream,"\nREAD INPUT FROM DISK\n");
	for( iter = tm.times_host_read.begin(); iter!=tm.times_host_read.end(); ++iter){
		fprintf_s(stream, "Elapsed time for %s : [%f]ms\n",iter->first,iter->second); 
		accum += iter->second;
	}
	fprintf_s(stream,"TOTAL ELAPSED TIME: [%f]\n",accum);	
	overall += accum; accum = 0;

	fprintf_s(stream,"\nWRITE OUTPUT TO DISK\n");
	for( iter = tm.times_host_write.begin(); iter!=tm.times_host_write.end(); ++iter){
		fprintf_s(stream, "Elapsed time for %s : [%f]ms\n",iter->first,iter->second); 
		accum += iter->second;
	}
	fprintf_s(stream,"TOTAL ELAPSED TIME: [%f]\n",accum);	
	overall += accum; accum = 0;

	fprintf_s(stream,"\n\nOVERALL ELAPSED TIME: [%f]\n",overall);	
	fprintf_s(stream,  "/***********************************************************/\n");
	fprintf_s(stream,  "/***********************************************************/\n");
	fflush(stream);
}

/**
  * Print device properties
  *
  * @param devProp  Device properties by CUDA
  */
void printDevProp(FILE *stream, cudaDeviceProp devProp)
{
	fprintf_s(stream,  "\n/***********************************************************/\n");
	fprintf_s(stream,    "/*********************DEVICE PROPERTIES*********************/\n");
	fprintf_s(stream,    "/***********************************************************/\n");
    fprintf_s(stream, "Major revision number:         %d\n",  devProp.major);
    fprintf_s(stream, "Minor revision number:         %d\n",  devProp.minor);
    fprintf_s(stream, "Name:                          %s\n",  devProp.name);
    fprintf_s(stream, "Total global memory:           %u\n",  devProp.totalGlobalMem);
    fprintf_s(stream, "Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
    fprintf_s(stream, "Total registers per block:     %d\n",  devProp.regsPerBlock);
    fprintf_s(stream, "Warp size:                     %d\n",  devProp.warpSize);
    fprintf_s(stream, "Maximum memory pitch:          %u\n",  devProp.memPitch);
    fprintf_s(stream, "Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
	fprintf_s(stream, "Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
	fprintf_s(stream, "Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);    
    fprintf_s(stream, "Total constant memory:         %u\n",  devProp.totalConstMem);
	for (int i = 0; i<2 ; ++i)
	fprintf_s(stream, "Maximum dim %d of texture:   %d\n", i, devProp.maxTexture2D[i]);    
    fprintf_s(stream, "Texture alignment:             %u\n",  devProp.textureAlignment);
    fprintf_s(stream, "Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    fprintf_s(stream, "Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    fprintf_s(stream, "Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	fprintf_s(stream,  "/***********************************************************/\n\n");
	fflush(stream);
    return;
}

/**
  * Calculates the optimal chunksize and other parameters that will be used.
  *
  * @param params  	Parameter structure for step3 calculations
  * @param meta 	Metadata for hyperparameters and stuff
  */
void calculate_parameters(PARAMETERS* params, METADATA* meta){

	// get device props 
	// TODO : assuming only 1 device
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp,  0); 
	//printDevProp(stdout, deviceProp);

	// calculate the number of pairwise corr 
	int n = meta->numSamples; 
	params->numCalc = (n*(n-1)/2);

	// calculate tile size
	// TODO calculate an optimal tile size for differing configurations
	// TODO an error checking must be added also
	params->tileSize = meta->tileSize;

	// number of passes that will be conducted along 1 dimension
	params->numPass = (meta->numSamples/meta->tileSize) + ( (meta->numSamples%meta->tileSize == 0) ? 0 : 1 );
	
	params->numTotalPass = (params->numPass*(params->numPass-1))/2 + params->numPass;
	meta->numChunks = params->numTotalPass; 
}

/**
  * Utility function for writing matrix chunk to file
  *
  * @param mat  	Matrix to be written
  * @param rows 	Rows of the matrix 
  * @param cols 	Columns of the matrix
  * @param i 		Idx of the current chunk
  * @param j 		Idy of the current chunk
  */
void write_matChunk_to_file(TYPE* mat, int rows, int cols, int i, int j, char* outdir){

	char filename[256];
	sprintf(filename, "%s/matChunk_%d_%d.txt" ,outdir,i,j);
	FILE* ftp = fopen(filename,"w+");
	
	for(int i=0; i<rows ; ++i,fprintf_s(ftp, "\n"))
		for(int j=0; j<cols ;++j,fprintf_s(ftp, " "))
			fprintf_s(ftp, "%f", mat[i*cols+j]);
	
	fclose(ftp);
}

/**
  * Utility function for writing matrix chunk to binary file
  *
  * @param mat  	Matrix to be written
  * @param rows 	Rows of the matrix 
  * @param cols 	Columns of the matrix
  * @param i 		Idx of the current chunk
  * @param j 		Idy of the current chunk
  */
void write_matChunk_to_binFile(TYPE* mat, int rows, int cols, int i, int j, char* outdir){
    
    FILE*   stream;      
    char    filename[256];
	sprintf(filename, "%s/matChunk_%d_%d.bin" ,outdir,i,j);

    SAFE_CALL(fopen_s(&stream, filename, "w+b"));
    
    if(fwrite((void*)mat,sizeof(TYPE),(size_t)rows*cols,stream) != (size_t)rows*cols){
        fprintf_s(stderr,"ERROR:%d-CANNOT WRITE FILE\n",__LINE__);	
        exit(-1);		
    }
    
    if(fclose(stream))
        fprintf_s(stderr, "The file was not closed\n" );

    if(_fcloseall())
        fprintf_s(stderr, "More than one file closed by _fcloseall\n");
}

/**
  * As the name refers but usees 'cudaError_t'
  *
  * @param message	
  */
void Check_CUDA_Error(const char *message)
{
   cudaError_t error = cudaGetLastError();
   if(error!=cudaSuccess) {
      fprintf_s(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
      exit(-1);
   }                         
}
