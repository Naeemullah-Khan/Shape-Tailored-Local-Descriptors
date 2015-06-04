#pragma once

#include "math.h"

#include <iostream>
#include <algorithm>
#include <vector>

#include "CImg.h"

#include "FastLabel_CV_mask.h"
#include "CG.h"

/*
#ifdef __MACH__
    // for mac (lion)
    #include "/opt/local/include/CImg.h"
#else
    // for linux (ubuntu)
    #include "/usr/include/CImg.h"
#endif
*/

#define T( U, r, c, d, nRow, nCol ) U[ (r) + (c) * nRow + (d) * nRow * nCol ]

#ifndef MAX 
    #define MAX( a, b ) ( ( (a) > (b) ) ? (a) : (b) )
#endif

#ifndef MIN
    #define MIN( a, b ) ( ( (a) < (b) ) ? (a) : (b) )
#endif

using namespace std;
using namespace cimg_library;

// *****************************************************************************
// This function performs a global active contour segmentation.
//
// [INPUT]
//      I       : (double) input image. (nRow x nCol)
//      L0      : (int) initial labels. (nRow x nCol)
//      M       : (int) mask. (nRow x nCol)
//      nLabel  : (int) number of labels
//      sigma   : (double) smoothing kernel size
//      nIter   : (int) number of interations
//      fPlot   : (int) flag for the display of intermediate steps
//
// [OUTPUT]
//      L       : (int) label map. (nRow x nCol)
//      R       : (double) reconstruction. (nRow x nCol)
//      E       : (int) energy. (1 x nIter)
//
//  E = \int_R G_sigma * F + length( \partial R )
//  F = ( I - cin ).^2 - ( I - cout ).^2
//
// [USAGE]
//  [ H, C, E ] = seg( double(I), uint32(L0), uint32(M), nLabel, sigma, nIter, fPlot );
// *****************************************************************************

template <class myType>
int WriteBinaryVector(myType* data,int size, std::string fileName)
{
	FILE* file = fopen (fileName.c_str(), "wb");
	if (file==NULL)
	{
		std::cout<<"Error: file for writing 3D image could not be open!"<<std::endl; 
		return -1;
	}
	fwrite(data, sizeof(myType), size, file);
	fclose(file);
	return 1;
}

int Initialize(std::vector<float*> I_big_vector_all,int* alpha_vector,int width, int height,int feature_vector_size,int** lables_init,int nLabel,double dSigma,int nIter,int nPlot,double *pL_out, double *pRecon_out, double *pErr_out)
{

	vector<float*> xSol_all;
	xSol_all.resize(feature_vector_size);
	float* xSol;
	for(int i=0;i<feature_vector_size;i++)
	{
		xSol=new float[height*width];
		memcpy(xSol,I_big_vector_all[i],(sizeof(float)*height*width));
		xSol_all[i]=xSol;
	}

	char* mask_char_vector=new char[width*height];
	memset(mask_char_vector,1,width*height*sizeof(char));
	
	
	int* edgebits_vector=new int[width*height];;
	memset(edgebits_vector,0,width*height*sizeof(int));	
	for (int p=0;p<width*height;p++)
	{
		if(p%height==0) edgebits_vector[p]+=2;
		if(p%(height)==height-1) edgebits_vector[p]+=1;
		if(p<height) edgebits_vector[p]+=8;
		if(p>(height*width)-height) edgebits_vector[p]+=4;
	}

	#pragma omp parallel for
	for(int k=0;k<feature_vector_size;k++)
	{
		ComputeCGSolution(1e-2f,xSol_all[k],I_big_vector_all[k],mask_char_vector,edgebits_vector,alpha_vector[k],width, height);
	}

	double* pI=new double[width*height*feature_vector_size];
	int count=0;
	for(int k=0;k<feature_vector_size;k++)
	{
		for(int i=0;i<width*height;i++)
		{
			pI[count]=xSol_all[k][i];
			count++;
		}
	}

	int* pM=new int[width*height];
	for(int i=0;i<width*height;i++)
	{
		pM[i]=1;
	}

	// Mask 1d vector for lables_init
	int* pL=new int[height*width];
	for (int i = 0; i < width; i++) 
	{
		for (int j = 0; j < height; j++) 
		{
			pL[(height*i)+j]=lables_init[j][i];
		}
	}

	memset(pL_out,0,width*height*sizeof(double));
	memset(pRecon_out,0,width*height*sizeof(double));
	memset(pErr_out,0,nIter*sizeof(int));

	int nRow=height;
	int nCol=width;
	int nDep=1;
	int nVec=feature_vector_size;
	//int nRowLabel;
	//int nColLabel;
	//int nDepLabel;
	//int nLabel;
	//int nIter;
	//int nPlot; 
	//double dSigma;
	WriteBinaryVector(pL,nRow*nCol,"pL_initial.bin");
	FastLabel_CV_mask(pI, pL, pM, pL_out, pRecon_out, pErr_out, nRow, nCol, nDep, nVec, nLabel, nIter, nPlot, dSigma);

	// Debug
    printf("nRow=%d, nCol=%d, nDep=%d, nVec=%d\n", nRow, nCol, nDep, nVec);
    printf("nLabel=%d, nIter=%d, nPlot=%d, dSigma=%d\n", nLabel, nIter, nPlot, dSigma);
    WriteBinaryVector(pI,nRow*nCol*nVec,"pI.bin");
    WriteBinaryVector(pL,nRow*nCol,"pL.bin");
    WriteBinaryVector(pM,nRow*nCol,"pM.bin");
    WriteBinaryVector(pL_out,nRow*nCol,"pL_out.bin");
    WriteBinaryVector(pRecon_out,nRow*nCol,"pRecon_out.bin");
    WriteBinaryVector(pErr_out,nIter,"pErr_out.bin");
    printf("pI=[%f,%f,%f]\n", pI[0], pI[1], pI[2]);

	return 1;
}

int FastLabel_CV_mask(double* pI, int* pL, int* pM, double *pL_out, double *pRecon_out, double *pErr_out, int nRow, int nCol, int nDep, int nVec, int nLabel, int nIter, int nPlot, double dSigma)
{
// 
//void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
//{
//    //const   mwSize *szImage, *szLabel, *szMask;
//    int     nDim, nRow, nCol, nDep, nVec;
//    int     nDimLabel, nRowLabel, nColLabel, nDepLabel;
//    int     nLabel = 2, nIter = 100, nPlot = 0;
//    double  dSigma = 1.6;
//
//    if( nrhs == 0 ) {
//        
//        mexErrMsgTxt( "Error: require an input data\n" );
//        return;
//    }
//
//    if( nrhs >= 1 ) {
//
//        szImage = mxGetDimensions( prhs[0] );
//        nDim    = mxGetNumberOfDimensions( prhs[0] );
//    }
//
//    if( nrhs >= 2 ) { 
//        
//        szLabel     = mxGetDimensions( prhs[1] ); 
//        nDimLabel   = mxGetNumberOfDimensions( prhs[1] );
//    }
//   
//    if( nrhs >= 3 ) { szMask    = mxGetDimensions( prhs[2] ); }
//    if( nrhs >= 4 ) { nLabel    = (int) mxGetScalar( prhs[3] ); }
//    if( nrhs >= 5 ) { dSigma    = (double) mxGetScalar( prhs[4] ); }
//    if( nrhs >= 6 ) { nIter     = (int) mxGetScalar( prhs[5] ); }
//    if( nrhs >= 7 ) { nPlot     = (int) mxGetScalar( prhs[6] ); }
//
//    nRow = szImage[0];
//    nCol = szImage[1];
//   
//    if( nDim == 2 ) {
//
//        nDep = 1;
//        nVec = 1;
//    }
//    else if( nDim == 3 ) {
//
//        if( nDim-1 == nDimLabel ) {
//
//            nDep = 1;
//            nVec = szImage[2];
//        }
//        else if( nDim == nDimLabel ) {
//
//            nDep = szImage[2];
//            nVec = 1;
//        }
//    }
//    else if( nDim == 4 ) {
//
//        nDep = szImage[2];
//        nVec = szImage[3];
//    }
//   
//    if( nPlot ) {
//
//        fprintf( stdout, "[input] dim: %d, row: %d, col: %d, dep: %d, vec: %d, label: %d, sigma: %2.2f, iter: %d\n", nDim, nRow, nCol, nDep, nVec, nLabel, dSigma, nIter, nPlot ); fflush( stdout );
//    }
//
//    const double *pI    = (double*) mxGetData( prhs[0] );
//    const int *pL       = (int*) mxGetData( prhs[1] );
//    const int *pM       = (int*) mxGetData( prhs[2] );
//
    CImg< double >      I( pI, nRow, nCol, nDep, nVec );
    CImg< int >         L( pL, nRow, nCol, nDep );
    CImg< int >         M( pM, nRow, nCol, nDep );
    CImg< int >         L_update( nRow, nCol, nDep );

    CImg< double >      F_blur( nRow, nCol, nDep );
    CImg< double >      F( nLabel, nRow, nCol, nDep );
    CImg< double >      FG( nLabel, nRow, nCol, nDep );

    int     *pAreaLabel             = new int[nLabel];
    double  **pConst                = new double*[nLabel];
    double  **pSumIntensityLabel    = new double*[nLabel];

    for( int p = 0; p < nLabel; p++ ) { 
        
        pConst[p]               = new double[nVec];
        pSumIntensityLabel[p]   = new double[nVec]; 
    }

    int     nMinIndex;
    double  dMinValue;
    int     *pErr   = new int[nIter];
    int     err     = 0;
    int     nIterStop = nIter;

    for( int i = 0; i < nIter; i++ ) { pErr[i] = -1; };

    pErr[0] = nRow * nCol * nDep * nVec;

    for( int i = 1; i < nIter; i++ ) {

        for( int p = 0; p < nLabel; p++ ) {

            pAreaLabel[p] = 0;
           
            for( int v = 0; v < nVec; v++ ) {

                pSumIntensityLabel[p][v] = 0;
            }
        }

        for( int r = 0; r < nRow; r++ ) {
            for( int c = 0; c < nCol; c++ ) {
                for( int d = 0; d < nDep; d++ ) {

                    if( M(r,c,d) ) {

                        pAreaLabel[L(r,c,d)] = pAreaLabel[L(r,c,d)] + 1;

                        for( int v = 0; v < nVec; v++ ) {

                            pSumIntensityLabel[L(r,c,d)][v] = pSumIntensityLabel[L(r,c,d)][v] + I(r,c,d,v);
                        }
                    }
                }
            }
        }

        for( int p = 0; p < nLabel; p++ ) {
        
            for( int v = 0; v < nVec; v++ ) {

                if( pAreaLabel[p] > 0 ) {
                
                    pConst[p][v] = pSumIntensityLabel[p][v] / pAreaLabel[p];
                }
                else {

                    pConst[p][v] = -100000;
                }
            }

            F.fill(0);
            F_blur.fill(0);

            for( int r = 0; r < nRow; r++ ) {
                for( int c = 0; c < nCol; c++ ) {
                    for( int d = 0; d < nDep; d++ ) {

                        if( M(r,c,d) ) {

                            for( int v = 0; v < nVec; v++ ) {

                                F(p,r,c,d) = F(p,r,c,d) + pow( I(r,c,d,v)-pConst[p][v], 2 );
                                F_blur(r,c,d) = F(p,r,c,d);
                            }
                        }
                    }
                }
            }

            if( nDep == 1 ) {

                F_blur.blur( dSigma, dSigma );
            }
            else {

                F_blur.blur( dSigma, dSigma, dSigma );
            }

            for( int r = 0; r < nRow; r++ ) {
                for( int c = 0; c < nCol; c++ ) {
                    for( int d = 0; d < nDep; d++ ) {
            
                        FG(p,r,c,d) = F_blur(r,c,d);
                    }
                }
            }
        }

        err = 0;
        L_update.fill(-1);

        for( int r = 0; r < nRow; r++ ) {
            for( int c = 0; c < nCol; c++ ) {
                for( int d = 0; d < nDep; d++ ) {

                    if( M(r,c,d) ) {
                        
                        nMinIndex = 0;
                        dMinValue = FG(0,r,c,d);

                        for( int p = 1; p < nLabel; p++ ) {

                            if( FG(p,r,c,d) < dMinValue ) {

                                dMinValue = FG(p,r,c,d);
                                nMinIndex = p;
                            }
                        }

                        L_update(r,c,d) = nMinIndex;

                        if( L_update(r,c,d) != L(r,c,d) ) {

                            err = err + 1;
                        }
                    }
                }
            }
        }

        L       = L_update;
        pErr[i] = err;

        if( nPlot ) {

            fprintf( stdout, "[%.4d] e = %d\n", i, pErr[i] ); fflush( stdout );
        }

        if( pErr[i] == 0 ) { nIterStop = i; break; }
    }
//
//    // output assignment.
//    int szPhase[2]  = { 1, nLabel };
//    int szError[2]  = { 1, nIter };
//    int szIter[2]   = { 1, 1 };
//    int nDimError   = 2;
//
//    plhs[0] = mxCreateNumericArray( nDimLabel, szLabel, (mxClassID) mxDOUBLE_CLASS, mxREAL );
//    plhs[1] = mxCreateNumericArray( nDimLabel, szLabel, (mxClassID) mxDOUBLE_CLASS, mxREAL );
//    plhs[2] = mxCreateNumericArray( nDimError, szError, (mxClassID) mxDOUBLE_CLASS, mxREAL );
//
//    double *pL_out      = (double*) mxGetData( plhs[0] );
//    double *pRecon_out  = (double*) mxGetData( plhs[1] );
//    double *pErr_out    = (double*) mxGetData( plhs[2] );
//
    for( int r = 0; r < nRow; r++ ) {
        for( int c = 0; c < nCol; c++ ) {
            for( int d = 0; d < nDep; d++ ) {

                T(pL_out,r,c,d,nRow,nCol)       = L(r,c,d);
                T(pRecon_out,r,c,d,nRow,nCol)   = 0;

                if( M(r,c,d) ) {
                
                    for( int v = 0; v < nVec; v++ ) {

                        T(pRecon_out,r,c,d,nRow,nCol) = T(pRecon_out,r,c,d,nRow,nCol) + pConst[L(r,c,d)][v]; 
                    }
                }
            }
        }
    }

    for( int ie = 0; ie < nIter; ie++ ) {

        pErr_out[ie] = pErr[ie];
    }

    for( int p = 0; p < nLabel; p++ ) { 
        
        delete [] pConst[p];
        delete [] pSumIntensityLabel[p]; 
    }
    
    delete [] pConst;
    delete [] pSumIntensityLabel;
    
    delete [] pErr;
    delete [] pAreaLabel;

	return 1;
}

