/*
 * globals.h
 *
 *  Description :   Global declarations and datatypes 
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

#ifndef GLOBALS_H_INCLUDED
#define GLOBALS_H_INCLUDED

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define TYPE float

#define BLOCK_X 16				/* kernel width	*/
#define BLOCK_Y 16				/* kernel heigh	*/
#define SHARED_BLOCKSIZE 256	/* pitch of shared memory length */

#define DEFAULT_MATRIX_CHUNKSIZE 4096	/* number of samples considered for each cublas operation also*/
                                        /* size of matrix size that correlation matrix will be calculated */

#define MAX_TEXTURE_WIDTH_IN_BYTES		65536	/*  */
#define MAX_TEXTURE_HEIGHT_IN_BYTES		32768	/*  */
#define MAX_PART_OF_FREE_MEMORY_USED	0.9		/*  */

typedef struct metaData{
	TYPE* data;
	int numSamples;
	int numTimesteps;
	int chunkSize;
    int tileSize;
	int _debug;
    int isOutputBinary;

	char* inputFile;
	char* outputDir;

	TYPE* d_sumX; 
	TYPE* h_sumX;

	TYPE* d_sumXX;
	TYPE* h_sumXX;

	TYPE** h_out_matrices;
	int numChunks;

}METADATA;

typedef struct parameters{

	int n;
	int numCalc;
	int numPass;
	int tileSize;
	int numTotalPass;

}PARAMETERS;

struct timer{

	std::map<char*, float> times1;
	std::map<char*, float> times2;
	std::map<char*, float> times3;

	std::map<char*, float> times_host_read;
	std::map<char*, float> times_host_write;

	cudaEvent_t start, stop; 
	float et;

};

#endif