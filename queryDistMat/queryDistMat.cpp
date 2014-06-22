/*
 * queryDistMat.cpp
 *
 *  Created on: 10.Mar.2012
 *      Author: orf
 *      E-mail: orhan.firat@ceng.metu.edu.tr
 *
 *      Vode An
 */

// If the code will be used in Matlab uncomment and compile with mex. Otherwise comment&compile
#define USE_MEX   

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef USE_MEX
	#include "mex.h"
    #include "matrix.h"
    #define SAFE_CALL(call)										\
     do {														\
       errno_t err = call;										\
       if (err != 0 ) {								    		\
       fprintf(stderr,"ERROR:%d-CANNOT OPEN FILE\n",__LINE__);	\
               return;											\
       }														\
     } while (false)
#else
    #include <getopt.h>
    #define SAFE_CALL(call)										\
     do {														\
       errno_t err = call;										\
       if (err != 0 ) {								    		\
       fprintf(stderr,"ERROR:%d-CANNOT OPEN FILE\n",__LINE__);	\
               exit(-1);										\
       }														\
     } while (false)
#endif

#define TYPE float

int _debug = 0;


/**
  * Utility function for reading matrix chunk from binary file
  *
  * @param mat  	
  * @param rows 	
  * @param cols 	
  * @param i 	
  * @param j 	
  */
void read_matChunk_from_binFile(TYPE* mat, int rows, int cols, int i, int j){

    FILE*   stream;      
    char    filename[128];    
	sprintf(filename, "matChunk_%d_%d.bin" ,i,j);

    SAFE_CALL(fopen_s(&stream, filename, "rb"));
    
    SAFE_CALL(fread(&mat[0], sizeof(TYPE), rows*cols, stream));
     
    if(fclose(stream))
        printf( "The file was not closed\n" );

    if(_fcloseall())
        printf( "More than one file closed by _fcloseall\n");
}

/**
  * Utility function for writing matrix chunk to binary file
  *
  * @param mat  	
  * @param rows 	
  * @param cols 	
  * @param i 	
  * @param j 	
  */
void write_matChunk_to_binFile(TYPE* mat, int rows, int cols, int i, int j){
    
    FILE*   stream;      
    char    filename[128];
	sprintf(filename, "matChunk_%d_%d.bin" ,i,j);

    SAFE_CALL(fopen_s(&stream, filename, "w+b"));
    
    if(fwrite((void*)mat,sizeof(TYPE),(size_t)rows*cols,stream) != (size_t)rows*cols){
        fprintf(stderr,"ERROR:%d-CANNOT WRITE FILE\n",__LINE__);	
        exit(-1);		
    }
    
    if(fclose(stream))
        fprintf(stderr, "The file was not closed\n" );

    if(_fcloseall())
        fprintf(stderr, "More than one file closed by _fcloseall\n");
}

/**
  * Utility function for writing result to binary file
  *
  * @param mat  	    Matrix to be written to file
  * @param rows 	    Number of rows in the output matrix
  * @param cols 	    Number of columns in the output matrix
  * @param outdir 	    Output directory 
  * @param queryRow     Index of the query row 	
  */
void write_mat_to_binFile(TYPE* mat, int rows, int cols, char* outdir, int queryRow){

    FILE*   stream;      
    char    filename[128];
	sprintf(filename, "%s/queryRowResult_%d.bin" ,outdir, queryRow);
    
    SAFE_CALL(fopen_s(&stream, filename, "w+b"));
    
    if (fwrite((void*)mat,sizeof(TYPE),(size_t)rows*cols,stream) != (size_t)rows*cols) {
        fprintf(stderr,"ERROR:%d-CANNOT WRITE FILE\n",__LINE__);	
        exit(-1);											
    }

    if(fclose(stream))
        fprintf(stderr, "The file was not closed\n" );

    if(_fcloseall())
        fprintf(stderr, "More than one file closed by _fcloseall\n");

}


/**
  * Subroutine of getDistMatRow(). Reads specified row from stream horizontaly or vertically.
  *
  * @param stream           Source stream for reading data	
  * @param resultVec        Destination vector for data read, should be preallocated	
  * @param queryInd         Query row index of the original numbering	
  * @param chunkSize        Chunk size of grid used in CUDA	
  * @param numRows          Number of rows to be considered, differs only on the last-row chunks
  * @param verticalFlag     Flag to indicate current chunk is in the upper or lower diagonal
  */
void fill_vector_with_binFile(FILE* stream,TYPE* resultVec,int queryInd,int chunkSize,int numRows,int verticalFlag){
    
    TYPE* safe_ptr = resultVec;        

    if(verticalFlag){
        for(int i=0; i<numRows ;++i){                        
            fseek(stream, (i*chunkSize+queryInd)*sizeof(TYPE) ,SEEK_SET);
            fread(&safe_ptr[i],sizeof(TYPE),1,stream); 
        }
    }else{        
        fseek(stream, chunkSize*queryInd*sizeof(TYPE) ,SEEK_SET);
        fread(safe_ptr, sizeof(TYPE), chunkSize, stream);
    }
}

/**
  * Interface to use CUDA code in Matlab (gateway routine).
  *
  * @param resultVec  	
  * @param numSamples 	
  * @param chunkSize 	
  * @param inputDir 	
  * @param queryRow 	
  */
void getDistMatRow(TYPE* resultVec, int numSamples, int chunkSize, int queryRow, const char* inputDir){

    FILE*   stream; 
    char    filename[128];

    int     x = 0;
    int     y = (queryRow/chunkSize);
    int     numChunks = (numSamples/chunkSize) + (numSamples%chunkSize==0 ? 0 : 1);
    int     swap = 0;

    for(int i=0 ; i<numChunks ; ++i, ( swap ? ++y : ++x)){
        
        // determine filename
        memset(filename, 0x0, sizeof(char)*128);
        sprintf(filename, "%s/matChunk_%d_%d.bin" ,inputDir ,y ,x);

        if(_debug)
            printf("reading file : %s\n",filename);

        // open current chunk file 
        SAFE_CALL(fopen_s(&stream, filename, "rb"));
             
        // read data to resultVec 
        fill_vector_with_binFile(stream, &resultVec[i*chunkSize], queryRow,chunkSize, 
            ((i+1)==numChunks ? numSamples%chunkSize : chunkSize ) , swap);

        fclose(stream);

        // determine upper diagonal to swap indices
        if(x==y)swap=1;
    }
}

#ifdef USE_MEX

/**
  * Interface to use CUDA code in Matlab (gateway routine).
  *
  * @param nlhs  	Number of expected mxArrays (Left Hand Side)
  * @param plhs 	Array of pointers to expected outputs
  * @param nrhs 	Number of inputs (Right Hand Side)
  * @param prhs 	Array of pointers to input data. The input data is read-only and should not be altered by your mexFunction .
  */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    
	// Variables
    int     numSamples;
    int     chunkSize;
    int     queryRow;
    char    inputDir[128];
    int     ref_width;
    TYPE*   resultVec;

    // Verification of the input arguments
    if( nrhs<5 || !mxIsChar(prhs[3]) )
        mexErrMsgTxt("Usage: resultVec = mexFunction(numSamples, chunkSize, queryRow, inputDir, debug);\n"
                     "numSamples  : total number of samples in the experiment (int)\n"
                     "chunkSize   : chunk size of grid used in CUDA (int)\n"
                     "queryRow    : query row index to be obtained (int)\n"
                     "inputDir    : folder containing input matrix chunk data (string)\n"
                     "debug       : 0 or 1 flag for output");              
    
	// Get input arguments
    numSamples  = (int)mxGetScalar(prhs[0]);
    chunkSize   = (int)mxGetScalar(prhs[1]);
    queryRow    = (int)mxGetScalar(prhs[2]);    
    _debug      = (int)mxGetScalar(prhs[4]);
    
    mxGetString(prhs[3], inputDir, mxGetN(prhs[3])+1);
    
    if(_debug){
        mexPrintf("There are %d right-hand-side argument(s).\n", nrhs);
        mexPrintf("numSamples   :%d\n", numSamples);
        mexPrintf("chunkSize    :%d\n", chunkSize);
        mexPrintf("queryRow     :%d\n", queryRow);
        mexPrintf("inputDir     :%s\n", inputDir);    
        mexPrintf("sizeof(TYPE) :%d\n", sizeof(TYPE));
    }

    // Allocation of output array
    if(sizeof(TYPE)==sizeof(float))
        resultVec = (TYPE *) mxGetPr(plhs[0] = mxCreateNumericMatrix(1, numSamples, mxSINGLE_CLASS, mxREAL));
    else
        resultVec = (TYPE *) mxGetPr(plhs[0] = mxCreateDoubleMatrix(1, numSamples, mxREAL));
    
    
    // obtain corresponding row from correlation matrix
    getDistMatRow(resultVec, numSamples, chunkSize, queryRow, inputDir);
    
}

#else // C code

static void usage(char *argv0) {
    char *help =
        "Usage: %s [switches] -q queryRow -c chunkSize -n numSamples\n"
        " -q queryRow   : query row index to be obtained\n"
        " -c chunkSize  : chunk size of grid used in CUDA\n"
        " -n numSamples : total number of samples in the experiment\n"
        " -f foldername : folder containing input matrix chunk data (default current dir)\n"
        " -o foldername : folder containing output query row (default current dir)\n"
        " -d : enable debug mode (default no)\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

int main(int argc, char **argv)
{
    // variables and parameters
    extern char*   optarg;
    extern int     optind;
    int     opt;
    int     chunkSize   = 4;
    int     numSamples  = 7;
    int     queryRow    = 2;
    char*   inputDir    = ".";  
    char*   outputDir   = ".";
 

    // process arguments
    while ( (opt=getopt(argc,argv,"q:c:n:f:dh?"))!= EOF) {
        switch (opt) {
            case 'q': queryRow=atoi(optarg);
                      break;
            case 'c': chunkSize=atoi(optarg);
                      break;
            case 'n': numSamples=atoi(optarg);
                      break;
            case 'f': inputDir=optarg;
                      break;
            case 'o': outputDir=optarg;
                      break;
            case 'd': _debug = 1;
                      break;
            case '?': usage(argv[0]);
                      break;
			case 'h': usage(argv[0]);
                      break;
            default: usage(argv[0]);
                      break;
        }
    }

    if (queryRow < 0 || queryRow > numSamples || numSamples==0 || chunkSize==0) usage(argv[0]); 

    // display informations
    if(_debug){
        printf("Number of samples       : %d\n", numSamples );
        printf("Chunk size used in CUDA : %d\n", chunkSize );
        printf("Index of query row      : %d\n", queryRow );
        printf("Input directory         : %s\n", inputDir );
        printf("Output directory        : %s\n", outputDir );
    }

    // allocate resulting vector
    TYPE* resultVec = (TYPE*)malloc(sizeof(TYPE)*numSamples);    
    memset(resultVec, 0x0, sizeof(TYPE)*numSamples);
    
    // obtain corresponding row from correlation matrix
    getDistMatRow(resultVec, numSamples, chunkSize, queryRow, inputDir);

    // write resulting vector to binary file 
    write_mat_to_binFile(resultVec, 1, numSamples, outputDir, queryRow);

    return 0;
}

#endif
