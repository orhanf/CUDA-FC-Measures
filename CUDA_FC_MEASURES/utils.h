/*
 * utils.h
 *
 *  Description :   Header for utils.cu 
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

#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <malloc.h>
#include <cublas.h>
#include <map>
#include "globals.h"

#define SAFE_CALL(call)										\
 do {														\
   errno_t err = call;										\
   if (err != 0 ) {								    		\
   fprintf_s(stderr,"ERROR:%d-CANNOT OPEN FILE\n",__LINE__);	\
           exit(-1);										\
   }														\
 } while (false)

TYPE* read_mat_from_file(const char*,int,int);

void print_matrix(TYPE*,int,int,int);

METADATA* init(int,char**);

void start_timer(void);

void set_host_timer(char* msg,float time, int step);

void stop_timer(char* msg,int step);

void print_timer(FILE *stream);

void printDevProp(FILE *stream, cudaDeviceProp devProp);

void calculate_parameters(PARAMETERS* params, METADATA* meta);

void write_matChunk_to_file(TYPE* mat, int rows, int cols, int i, int j, char* outdir);

void write_matChunk_to_binFile(TYPE* mat, int rows, int cols, int i, int j, char* outdir);

void Check_CUDA_Error(const char *message);

void usage(void);

#endif