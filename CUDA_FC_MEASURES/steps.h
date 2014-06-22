/*
 * steps.h
 *
 *  Description :   Header for steps.cu 
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

#ifndef STEPS_H_INCLUDED
#define STEPS_H_INCLUDED

#include <cublas.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include "globals.h"
#include "utils.h"

#define CUDA_SAFE_CALL(call)																\
 do {																						\
    cudaError_t error = call;																\
   if (error != cudaSuccess ) {																\
         fprintf_s(stderr, "CUDA ERROR:%s in line %d\n",cudaGetErrorString(error),__LINE__);	\
         exit(-1);																			\
   }																						\
 } while (false)


/**
  * Funtion to calculate sum of individual time series by a matrix-vector multiplication using cublas.
  *
  * @param meta		Metadata structure for hyper parameters and stuff		
  */
void step1_calculate_sumX(METADATA* meta);

/**
  *	Funtion to calculate sum of squares of individual time series by a matrix-vector 
  * multiplication using cublas. Square operation conducted using thrust library.
  * Written as a seperate function with almost the same code except thrust routines,
  * step1 and step2 are available for fully parallelisation for multiple GPUs 
  *	or the ones that can launch concurrent kernels.
  *
  * @param meta		Metadata structure for hyper parameters and stuff		
  */
void step2_calculate_sumXX(METADATA* meta);

/**
  * Major function to calculate pearson-correlation coefficient using previous
  * steps' results on device.
  *
  * @param meta		Metadata structure for hyper parameters and stuff		
  */
void step3_calculate_pearson_corr(METADATA* meta);

#endif
