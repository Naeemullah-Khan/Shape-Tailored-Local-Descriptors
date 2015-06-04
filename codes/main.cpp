#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <fstream>
#include <string>
#include <vector>
#include "timer.h"
#include "CG.h"
#include "ConjugateGradient.h"
#include "Params.h"
#include "DirectionalGradient.h"
#include "Utilities.h"
#include "FastLabel_CV_mask.h"

using namespace cv;
using namespace std;

//#define CONTOUR_INIT
#define CONTOUR_DISPLAY
#define ITER_COUNT


int main(int argc, char *argv[])
{

	char* imageIn;
	char* resultOut;
	string baseFolder = "";
	if (argc == 3)
	{
		imageIn = argv[1];
		resultOut = argv[2];
		cout << "image:" << imageIn << endl;
		cout << "result:" << resultOut << endl;
	}
	else
	{
		cout << "number of arguments is not correct." << endl;
		return -1;
	}

	// Read image
	std::ofstream myfile;

	timer timer_data_read;
	timer_data_read.tic();
	Mat im = imread(imageIn, CV_LOAD_IMAGE_GRAYSCALE);
	if (im.empty())
	{
		cout << "Cannot load image!" << endl;
		return -1;
	}


	// Resize
	int width=256,height=256;
	cv::resize(im,im, cv::Size(width,height));

	// Read image to 2d float
	float** image2d=new float*[im.rows];
	for(int i = 0; i < im.rows; ++i) image2d[i] = new float[im.cols];
	for (int i = 0; i < im.cols; i++) 
	{
		for (int j = 0; j < im.rows; j++) 
		{
			image2d[i][j]=(float)im.data[im.step[0]*i + im.step[1]* j];
		}
	}
	
	// Read mask to 2d float
	int** mask2d=new int*[im.rows];
	for(int i = 0; i < im.rows; ++i) mask2d[i] = new int[im.cols];
	for (int i = 0; i < im.cols; i++) 
	{
		for (int j = 0; j < im.rows; j++) 
		{
			mask2d[i][j]=0;
		}
	}






		for (int i =10; i < 45; i++) 
	{
		for (int j = 10; j < 45; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 60; j < 95; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 110; j < 145; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 160; j < 195; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 210; j < 245; j++) 
		{
			mask2d[i][j]=1;
		}
	}

	for (int i =60; i < 95; i++) 
	{
		for (int j = 10; j < 45; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 60; j < 95; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 110; j < 145; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 160; j < 195; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 210; j < 245; j++) 
		{
			mask2d[i][j]=1;
		}
	}
		
	for (int i =110; i < 145; i++) 
	{
		for (int j = 10; j < 45; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 60; j < 95; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 110; j < 145; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 160; j < 195; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 210; j < 245; j++) 
		{
			mask2d[i][j]=1;
		}
	}


	for (int i =160; i < 195; i++) 
	{
		for (int j = 10; j < 45; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 60; j < 95; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 110; j < 145; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 160; j < 195; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 210; j < 245; j++) 
		{
			mask2d[i][j]=1;
		}
	}
		
	for (int i =210; i < 245; i++) 
	{
		for (int j = 10; j < 45; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 60; j < 95; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 110; j < 145; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 160; j < 195; j++) 
		{
			mask2d[i][j]=1;
		}

		for (int j = 210; j < 245; j++) 
		{
			mask2d[i][j]=1;
		}
	}


	// Make 2d edgebits 
	int** edgebits2d=new int*[height];
	for(int i = 0; i < height; ++i) edgebits2d[i] = new int[width];

	for (int i = 0; i < height; i++) 
	{
		for (int j = 0; j < width; j++) 
		{
			edgebits2d[i][j]=0;
			if(i==0) edgebits2d[i][j]+=2;
			if(i==height-1) edgebits2d[i][j]+=1;
			if(j==0) edgebits2d[i][j]+=8;
			if(j==width-1) edgebits2d[i][j]+=4;
			
			if(i>0&&i<height-1&&j>0&&j<width-1)
			{
				if(mask2d[i][j]>0&&mask2d[i+1][j]==0)edgebits2d[i][j]+=1;
				if(mask2d[i][j]>0&&mask2d[i-1][j]==0)edgebits2d[i][j]+=2;
				if(mask2d[i][j]>0&&mask2d[i][j+1]==0)edgebits2d[i][j]+=4;
				if(mask2d[i][j]>0&&mask2d[i][j-1]==0)edgebits2d[i][j]+=8;
			}

		}
	}

	// Compute phi2d
	float** phi2d=new float*[height];
	for(int i = 0; i < height; ++i) phi2d[i] = new float[width];
	mask2phi(mask2d,width,height,phi2d);

	printf("Data Reading timer: %f\n", ((timer_data_read.toc()/1000)/60) );

	// Compute Directional Gradient
	float** gx=new float*[height];
	for(int i = 0; i < height; ++i) gx[i] = new float[width];
	
	float** gy=new float*[height];
	for(int i = 0; i < height; ++i) gy[i] = new float[width];
	
	float*** im_grad_matrix=new float**[height];
	for(int i = 0; i < height; ++i) im_grad_matrix[i] = new float*[width];
	for(int i = 0; i < height; ++i) for(int j = 0; j < width; ++j) im_grad_matrix[i][j] = new float[step_theeta];
	
	int bins_num=(int)(step_theeta/bins_theeta/overlap_factor);
	int** bin_matrix=new int*[bins_theeta];
	for(int i = 0; i < bins_theeta; ++i) bin_matrix[i] = new int[bins_num];

	timer timer_directional_gradient;
	timer_directional_gradient.tic();
	ComputeDirectionalGradient(image2d,mask2d,width,height,bins_num,gx,gy,im_grad_matrix,bin_matrix);	
	printf("Directional Gradient timer: %f\n", ((timer_directional_gradient.toc()/1000)/60) );

	// Make 1d image vector
	float* image_vector=new float[height*width];
	for (int i = 0; i < width; i++) 
	{
		for (int j = 0; j < height; j++) 
		{
			image_vector[(height*i)+j]=image2d[j][i];
		}
	}

	// Mask 1d vector for mask
	int* mask_vector=new int[height*width];
	for (int i = 0; i < width; i++) 
	{
		for (int j = 0; j < height; j++) 
		{
			mask_vector[(height*i)+j]=mask2d[j][i];
		}
	}

	// Mask 1d vector for mask
	char* mask_char_vector=new char[height*width];
	for (int i = 0; i < width*height; i++) 
	{
		mask_char_vector[i]=(int)mask_vector[i];
	}

	// Make edgebits 1d vector
	int* edgebits_vector=new int[height*width];
	for (int i = 0; i < width; i++) 
	{
		for (int j = 0; j < height; j++) 
		{
			edgebits_vector[(height*i)+j]=edgebits2d[j][i];
		}
	}

	// Make 1d phi vector
	float* phi_vector=new float[height*width];
	for (int i = 0; i < width; i++) 
	{
		for (int j = 0; j < height; j++) 
		{
			phi_vector[(height*i)+j]=phi2d[j][i];
		}
	}

	// generting RHS for Poisson PDE
	timer timer_rhs;
	timer_rhs.tic();
	int feature_vector_size=num_a*(bins_theeta+1);
	int* alpha_vector=new int[feature_vector_size];

	vector<float*> I_big_vector_all;
	I_big_vector_all.resize(feature_vector_size);
	float* I_big_vector;	
	for(int i=0;i<feature_vector_size;i++)
	{
		I_big_vector=new float[height*width];
		I_big_vector_all[i]=I_big_vector;
	}

	int alpha;
	float sum;
	int k=0,bin;
	for(int i=0;i<=num_a-1;i++)
	{
		alpha=a0+(step_a*i);
		for(int j=0;j<=bins_theeta;j++)
		{
			if(j==0)
			{
				for(int ii=0;ii<height;ii++)
				{
					for(int jj=0;jj<width;jj++)
					{
						I_big_vector_all[k][(height*jj)+ii]=0.5f*image2d[ii][jj];
					}
				}
				alpha_vector[k]=alpha;
				k++;
			}
			else
			{
				for(int ii=0;ii<height;ii++)
				{
					for(int jj=0;jj<width;jj++)
					{
						sum=0;						
						for(int kk=0;kk<bins_num;kk++)
						{
							bin=bin_matrix[j-1][kk];
							sum+=im_grad_matrix[ii][jj][bin];
						}
						I_big_vector_all[k][(height*jj)+ii]=sum;
					}
				}
				alpha_vector[k]=alpha;
				k++;
			}
		}
	}
	printf("RHS Generation timer: %f\n", ((timer_rhs.toc()/1000)/60) );

#ifdef CONTOUR_INIT

	// Initialize
	int nLabel=2;
	double sigma   = 1.2;    // regularization.
	int nIter   = 100;  // number of iterations.
	int nPlot   = 1;    // flag for displaying intermediate results.
    double *pL_out      = new double[width*height];
    double *pRecon_out  = new double[width*height];
    double *pErr_out    = new double[nIter];
 
	//L0(40:200,40:200)=1;
	int** lables_init=new int*[height];
 	for(int i = 0; i < height; ++i) lables_init[i] = new int[width];
	 for (int i = 0; i < width; i++) 
 	{
		 for (int j = 0; j < height; j++) 
		 {
			 lables_init[i][j]=0;
 		}
	 }

	 for (int i = 60; i < 200; i++) 
	 {
		for (int j = 60; j < 250; j++) 
		 {
			 lables_init[i][j]=1;
 		}
	 }

	timer timer_init;
	timer_init.tic();
 	Initialize(I_big_vector_all,alpha_vector,width, height,feature_vector_size,lables_init,nLabel,sigma,nIter,nPlot,pL_out, pRecon_out, pErr_out);
	printf("Initialization timer: %f\n", (timer_init.toc()/1000) );

#endif

	// Conjugate Gradient

	timer timer_cg;
	timer_cg.tic();
	
	float Energy=0;
	float Energy_old=0;
	float mul_factor=1;
	float* u;
	float* v;
	float* u_hat;
	float* v_hat;
	float* u_hat_rhs;
	float* v_hat_rhs;
	float* kappa=new float[height*width]; 
	float* gxi1=new float[height*width]; 
	float* gxi2=new float[height*width]; 
	float* gyi1=new float[height*width]; 
	float* gyi2=new float[height*width]; 
	float* gxo1=new float[height*width]; 
	float* gxo2=new float[height*width]; 
	float* gyo1=new float[height*width]; 
	float* gyo2=new float[height*width]; 
	string Result;

	vector<float*> u_all;
	vector<float*> v_all;
	vector<float*> u_hat_all;
	vector<float*> v_hat_all;
	vector<float*> u_hat_rhs_all;
	vector<float*> v_hat_rhs_all;

	u_all.resize(feature_vector_size);
	v_all.resize(feature_vector_size);
	u_hat_all.resize(feature_vector_size);
	v_hat_all.resize(feature_vector_size);
	u_hat_rhs_all.resize(feature_vector_size);
	v_hat_rhs_all.resize(feature_vector_size);

	for(int i=0;i<feature_vector_size;i++)
	{
		u=new float[height*width];
		memcpy(u,I_big_vector_all[i],(sizeof(float)*height*width));
		u_all[i]=u;
	}

	for(int i=0;i<feature_vector_size;i++)
	{
		v=new float[height*width];
		memcpy(v,I_big_vector_all[i],(sizeof(float)*height*width));
		v_all[i]=v;
	}

	for(int i=0;i<feature_vector_size;i++)
	{
		u_hat=new float[height*width];
		memcpy(u_hat,I_big_vector_all[i],(sizeof(float)*height*width));
		u_hat_all[i]=u_hat;
	}

	for(int i=0;i<feature_vector_size;i++)
	{
		v_hat=new float[height*width];
		memcpy(v_hat,I_big_vector_all[i],(sizeof(float)*height*width));
		v_hat_all[i]=v_hat;
	}

	for(int i=0;i<feature_vector_size;i++)
	{
		u_hat_rhs=new float[height*width];
		memcpy(u_hat_rhs,I_big_vector_all[i],(sizeof(float)*height*width));
		u_hat_rhs_all[i]=u_hat_rhs;
	}

	for(int i=0;i<feature_vector_size;i++)
	{
		v_hat_rhs=new float[height*width];
		memcpy(v_hat_rhs,I_big_vector_all[i],(sizeof(float)*height*width));
		v_hat_rhs_all[i]=v_hat_rhs;
	}

	float* ui=new float[feature_vector_size];
	float* vi=new float[feature_vector_size];

	char* mask_b=new char[height*width];
	char* mask_in=new char[height*width];
	char* mask_out=new char[height*width];

	int* edgebits_in=new int[height*width];
	int* edgebits_out=new int[height*width];

	float* f=new float[height*width];

	#ifdef CONTOUR_DISPLAY
	int contour_row,contour_column;
	namedWindow("contour_window1", CV_WINDOW_NORMAL);
	Mat contour_image1 = imread(baseFolder + imageIn, CV_LOAD_IMAGE_COLOR);
	cv::resize(contour_image1,contour_image1, cv::Size(width,height));
	Mat contour_image2 = imread(baseFolder + imageIn, CV_LOAD_IMAGE_COLOR);
	contour_image1.copyTo(contour_image2);
	imshow("contour_window1", contour_image2);
	waitKey(1);
	#endif

	
	int max_its=1000;
	float lowerVal=-1.5f; //limit for levelset slack
	float upperVal=1.5f;  //limit for levelset slack
	int count_in=0,count_out=0;
	for(int itr=0;itr<max_its;itr++)
	{
		//mask_b is narrowband mask_in & mask_out are inside and outside regions ..edgebits are label for pixel on the boundary of respective regions
		phi2mask3(phi_vector,height,width,mask_b,mask_in,mask_out,edgebits_in,edgebits_out,lowerVal, upperVal);
	


		#pragma omp parallel for
		for(int k=0;k<feature_vector_size;k++)
		{
			//returns the descriptor u inside the region
			ComputeCGSolution(1e-2f,u_all[k],I_big_vector_all[k],mask_in,edgebits_in,alpha_vector[k],width, height);
		}
		count_in=0;
		count_out=0;
		for(int p=0;p<(height*width);p++)
		{
			if(mask_in[p]==1) count_in++;
			if(mask_out[p]==1) count_out++;
		}

		#pragma omp parallel for
		for(int k=0;k<feature_vector_size;k++)
		{
			ui[k]=0; //count=0;
			for(int p=0;p<(height*width);p++)
			{
				if(mask_in[p]==1) 
				{
						//average value of u
						ui[k]+=u_all[k][p];
						//count++;
				}
			}
			ui[k]=ui[k]/count_in;
			for(int p=0;p<(height*width);p++)
			{
				if(mask_in[p]==1) 
				{		
						//rhs for u_hat descriptor
						u_hat_rhs_all[k][p]=(u_all[k][p]-ui[k]);   //*(u_all[k][p]-ui[k])
				}
			}
		}
		
		#pragma omp parallel for
		for(int k=0;k<feature_vector_size;k++)
		{
			//descritor u hat
			ComputeCGSolution(1e-2f,u_hat_all[k],u_hat_rhs_all[k],mask_in,edgebits_in,alpha_vector[k],width, height);
		}
				
		#pragma omp parallel for
		for(int k=0;k<feature_vector_size;k++)
		{
			// returns descriptor v
			ComputeCGSolution(1e-2f,v_all[k],I_big_vector_all[k],mask_out,edgebits_out,alpha_vector[k],width, height);
		}

		#pragma omp parallel for
		for(int k=0;k<feature_vector_size;k++)
		{
			vi[k]=0; //count=0;
			for(int p=0;p<(height*width);p++)
			{
				if(mask_out[p]==1) 
				{		//average value of v
						vi[k]+=v_all[k][p];
						//count++;
				}
			}
			vi[k]=vi[k]/count_out;
			for(int p=0;p<(height*width);p++)
			{
				if(mask_out[p]==1) 
				{
						//rhs of v hat
						v_hat_rhs_all[k][p]=(v_all[k][p]-vi[k]);    //*(v_all[k][p]-vi[k])
				}
			}
		}
		//WriteVector(vi,feature_vector_size,baseFolder+"vec2.txt");
		#pragma omp parallel for
		for(int k=0;k<feature_vector_size;k++)
		{	
			//descriptor v hat
			ComputeCGSolution(1e-2f,v_hat_all[k],v_hat_rhs_all[k],mask_out,edgebits_out,alpha_vector[k],width, height);
		}


		//calculate force on narrowband (mask_b)
		//F 
		if (itr%10==0)
		PlotDescriptor( u_all,v_all,height,width);
		memset(f,0,sizeof(float)*height*width);
		for(int k=0;k<feature_vector_size;k++)
		{
			#pragma omp parallel sections
			{	
				//gradients for all descriptors u,v, uhat vhat
				#pragma omp section
				{
					ComputeGradient( u_all[k] ,mask_in,edgebits_in,gxi1,gyi1,height*width,height);
				}
				#pragma omp section
				{
					ComputeGradient( u_hat_all[k] ,mask_in,edgebits_in,gxi2,gyi2,height*width,height);
				}
				#pragma omp section
				{
					ComputeGradient( v_all[k] ,mask_out,edgebits_out,gxo1,gyo1,height*width,height);
				}
				#pragma omp section
				{
					ComputeGradient( v_hat_all[k] ,mask_out,edgebits_out,gxo2,gyo2,height*width,height);
				}
			}
			for (int p=0 ;p<height*width;p++)
			{	

				if (mask_b[p]==1)
				{	//energy term for pixels on narrow band
					f[p]+=((u_all[k][p]-ui[k])*(u_all[k][p]-ui[k])-(v_all[k][p]-vi[k])*(v_all[k][p]-vi[k]));
					f[p]+=(2/alpha_vector[k])*(u_hat_all[k][p]*(u_all[k][p]-I_big_vector_all[k][p]));
					f[p]-=(2/alpha_vector[k])*(v_hat_all[k][p]*(v_all[k][p]-I_big_vector_all[k][p]));
					f[p]+=2*(gxi1[p]*gxi2[p]+gyi1[p]*gyi2[p]-gxo1[p]*gxo2[p]-gyo1[p]*gyo2[p])/alpha_vector[k];

				}
			}
		}
		//WriteVector(f,height*width,baseFolder+"f.txt");
		//calculate curvature flow on the boundary
		ComputeKappa(phi_vector ,kappa, mask_b, height*width,height);
		float max_f=VecMax(f,height*width);
		for (int p=0 ;p<height*width;p++)
		{	//normalization
			f[p]=f[p]/max_f;
			f[p]+=(0.75f*kappa[p]);
			
		}
		//WriteVector(f,height*width,baseFolder+"f.txt");	
		max_f=VecMax(f,height*width);
		for (int p=0 ;p<height*width;p++)
		{ 
			//level set update
			phi_vector[p]+=f[p]*mul_factor*0.45f/(max_f+0.0001f);
			
		}
		//smoothing of the levelset
		Sussman(phi_vector,0.5,height*width,height);ostringstream convert;   // stream used for the conversion

		convert << itr;      // insert the textual representation of 'Number' in the characters in the stream

		Result = convert.str();
		#ifdef ITER_COUNT
		cout<<"it count"<<max_f<<endl<<itr<<endl;
		#endif
		if (itr>200 && itr%10==0)
		{	
			Energy=0;
			for (int k=0;k<feature_vector_size;k++)
			{	for (int p=0;p<height*width;p++)
				{
					if (phi_vector[p]>0)
						Energy+=(v_all[k][p]-vi[k])*(v_all[k][p]-vi[k]);
					else
						Energy+=(u_all[k][p]-ui[k])*(u_all[k][p]-ui[k]);
				}
			}		
		}
		if (itr==210) Energy_old=Energy;
		if (Energy>1.1*Energy_old)
		{ 
			mul_factor/=2; 
			cout<<"energy old="<<Energy_old<<"   energy new="<<Energy<<endl;
			cout<<"mul_factor="<<mul_factor;
			if (mul_factor<0.05)
			{
				cout<<"energy increasing"<<endl; break;
			}
		}
		Energy_old=Energy;


		#ifdef CONTOUR_DISPLAY
		// Display
		if (itr%20==0)
		{
			contour_image1.copyTo(contour_image2);
			for (int i = 0; i < height*width; i++) 
			{
				if(phi_vector[i]>=-1&&phi_vector[i]<=1)
				{
					contour_row=(i%height);
					contour_column=(int) i/height;
					if(contour_row<0||contour_row>=height||contour_column<0||contour_column>=width);
					else
					{
						contour_image2.at<cv::Vec3b>(contour_row,contour_column)[0] = 255;
						contour_image2.at<cv::Vec3b>(contour_row,contour_column)[1] = 0;
						contour_image2.at<cv::Vec3b>(contour_row,contour_column)[2] = 0;
					}
				}
			}
			imshow("contour_window", contour_image2);
			waitKey(1);
		}
		#endif
	}
	


	printf("Conjugate Gradient timer: %f\n", ((timer_cg.toc()/1000)) );
	WriteBinaryVector3d(I_big_vector_all,feature_vector_size,height,width,(baseFolder+"I_big_vector_all.bin"));
	WriteBinaryVector3d(u_all,feature_vector_size,height,width,(baseFolder+"u_all.bin"));
	WriteBinaryVector3d(v_all,feature_vector_size,height,width,(baseFolder+"v_all.bin"));
	WriteBinaryVector3d(u_hat_all,feature_vector_size,height,width,(baseFolder+"u_hat_all.bin"));
	WriteBinaryVector3d(v_hat_all,feature_vector_size,height,width,(baseFolder+"v_hat_all.bin"));
	WriteVector(vi,feature_vector_size,(baseFolder+"vi.txt"));
	WriteVector(ui,feature_vector_size,(baseFolder+"ui.txt"));

	WriteVector(f,height*width,baseFolder+"f.txt");
	WriteVector(phi_vector,height*width,baseFolder+"phi_vector.txt");
	WriteVector(mask_in,(height*width),(baseFolder+"mask_in.txt"));		
	WriteVector(mask_out,(height*width),(baseFolder+"mask_out.txt"));
	WriteVector(mask_b,(height*width),(baseFolder+"mask_b.txt"));
	WriteVector(gxi1,height*width,baseFolder+"gxi1.txt");
	WriteVector(gyi1,height*width,baseFolder+"gyi1.txt");
	WriteVector(gxi2,height*width,baseFolder+"gxi2.txt");
	WriteVector(gyi2,height*width,baseFolder+"gyi2.txt");
	WriteVector(gxo1,height*width,baseFolder+"gxo1.txt");
	WriteVector(gyo1,height*width,baseFolder+"gyo1.txt");
	WriteVector(gxo2,height*width,baseFolder+"gxo2.txt");
	WriteVector(gyo2,height*width,baseFolder+"gyo2.txt");
	
	cin.ignore();

	return 1;

}