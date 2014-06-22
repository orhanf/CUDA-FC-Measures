/*
 * main.cpp
 *
 *  Description :   main 
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


#include <cuda.h>
#include <time.h>
#include "utils.h"
#include "steps.h"

using namespace std;

METADATA* META;

int main(int argc,char **argv)
{
		
	META = init(argc,argv);
	
	step1_calculate_sumX(META);

	step2_calculate_sumXX(META);
	
	step3_calculate_pearson_corr(META);

	print_timer(stdout);

	return 0;
}
