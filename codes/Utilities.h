#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

template <class myType>
Mat* ImageToGrayMat(myType** image2d,int width,int height)
{
	Mat* im=new Mat(height,width, CV_8UC1);
	for (int i = 0; i < im->cols; i++) 
	{
		for (int j = 0; j < im->rows; j++) 
		{
			im->data[im->step[0]*i + im->step[1]* j]=(uchar)image2d[i][j];
		}
	}
	return im;
}

template <class myType>
Mat* ImageToFloatMat(myType** image2d,int width,int height)
{
	Mat* im=new Mat(height,width, CV_32FC1);
	for (int i = 0; i < im->cols; i++) 
	{
		for (int j = 0; j < im->rows; j++) 
		{
			im->data[im->step[0]*i + im->step[1]* j]=(uchar)image2d[i][j];
		}
	}
	return im;
}

template <class myType>
Mat* ImageToGrayMat(myType* image1d,int width,int height)
{
	Mat* im=new Mat(height,width, CV_8UC1);
	int counter=0;
	for (int i = 0; i < im->cols; i++) 
	{
		for (int j = 0; j < im->rows; j++) 
		{
			im->data[(im->cols*j) + i]=(uchar)image1d[counter++];
		}
	}
	return im;
}

string IntToStr(int number)
{
   stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

int mask2phi(int** mask2d,int width,int height, float** phi)
{
	Mat mask=*ImageToGrayMat(mask2d,width,height);
	Mat maskf=*ImageToFloatMat(mask2d,width,height);
	Mat maskInv=1-mask;

	Mat phi1,phi2;
	distanceTransform(mask,phi1,CV_DIST_L2,3);
	distanceTransform(maskInv,phi2,CV_DIST_L2,3);
	Mat phi3=phi2-phi1;
	Mat phiMat=phi3+maskf;
	phiMat=phiMat-0.5;
   

	// phiMat to phi
	for (int i = 0; i < phiMat.cols; i++) 
	{
		for (int j = 0; j < phiMat.rows; j++) 
		{
			phi[i][j]=(float)phiMat.at<float>(Point(j,i));
		}
	}



	return 1;
}

inline int phi2mask3(float* phi,int width,int height,char* mask_b,char* mask_in,char* mask_out, int* edgebits_in, int* edgebits_out,float lowerVal, float upperVal)
{
	int length=width*height;
	memset(mask_b,0,length*sizeof(char));
	memset(mask_in,0,length*sizeof(char));
	memset(mask_out,0,length*sizeof(char));
	memset(edgebits_in,0,length*sizeof(int));
	memset(edgebits_out,0,length*sizeof(int));
	for(int i=0;i<length;i++)
	{
		if(phi[i]>=lowerVal&&phi[i]<=upperVal) mask_b[i]=1;
		if(phi[i]>lowerVal) mask_out[i]=1;
		if(phi[i]<upperVal) mask_in[i]=1;
	}
	
	// Make 1d edgebits_in 

	for (int i = 0; i < length; i++) 
	{
			
			if(i%height==0) edgebits_in[i]+=2;
			if(i%(height)==height-1) edgebits_in[i]+=1;
			if(i<height) edgebits_in[i]+=8;
			if(i>length-height-1) edgebits_in[i]+=4;
			
			if(i>=height&&i<=length-height&&i%(height)!=height-1&&i%height!=0)
			{
				if(mask_in[i]>0&&mask_in[i+1]==0)edgebits_in[i]+=1;
				if(mask_in[i]>0&&mask_in[i-1]==0)edgebits_in[i]+=2;
				if(mask_in[i]>0&&mask_in[i+height]==0)edgebits_in[i]+=4;
				if(mask_in[i]>0&&mask_in[i-height]==0)edgebits_in[i]+=8;
			}		
	}

	// Make 1d edgebits_out 
	

	for (int i = 0; i < length; i++) 
	{
			
			if(i%height==0) edgebits_out[i]+=2;
			if(i%(height)==height-1) edgebits_out[i]+=1;
			if(i<height) edgebits_out[i]+=8;
			if(i>length-height-1) edgebits_out[i]+=4;
			
			if(i>=height&&i<=length-height&&i%(height)!=height-1&&i%height!=0)
			{
				if(mask_out[i]>0&&mask_out[i+1]==0)edgebits_out[i]+=1;
				if(mask_out[i]>0&&mask_out[i-1]==0)edgebits_out[i]+=2;
				if(mask_out[i]>0&&mask_out[i+height]==0)edgebits_out[i]+=4;
				if(mask_out[i]>0&&mask_out[i-height]==0)edgebits_out[i]+=8;
			}		
	}

	return 1;
}

template <class myType>
inline myType VecMax(myType* vec,int length)
{
	myType max_val=vec[0];
	for(int i=0;i<length;i++)
	{
		if (vec[i]>0&&vec[i]>max_val) max_val=vec[i];
		else if(vec[i]<0 &&-vec[i]>max_val) max_val=-vec[i];
	}
	if (max_val>0)
	return max_val;
	else if(max_val==0)
	return max_val=1;
	
	return max_val=1;
}

template <class myType>
int WriteVector(myType* vec,int length,string file_name)
{
	// write 1d vector
	std::ofstream myfile;
	myfile.open(file_name);
	for (int i = 0; i < length; i++) 
	{
		myfile<<(double)vec[i]<<endl;
	}
	myfile.close();

	return 1;
}

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

int WriteBinaryVector3d(vector<float*> data,int feature_vector_size,int height,int width,std::string fileName)
{
	float* data2=new float[feature_vector_size*height*width];
	int dataCounter2=0;
	for(int i=0;i<feature_vector_size;i++)
	{
		for(int j=0;j<height*width;j++)
		{
			data2[dataCounter2++]=data[i][j];
		}
	}

	FILE* file = fopen (fileName.c_str(), "wb");
	if (file==NULL)
	{
		std::cout<<"Error: file for writing 3D image could not be open!"<<std::endl; 
		return -1;
	}
	fwrite(data2, sizeof(float), (feature_vector_size*height*width), file);
	fclose(file);
	return 1;
}



int Sussman(float* phi,float dt,int length,int height)
{
	float* a;
	float* b;
	float* c;
	float* d;
	float* a_p;
	float* b_p;
	float* c_p;
	float* d_p;
	float* a_n;
	float* b_n;
	float* c_n;
	float* d_n;
	float* delta_phi;
	float val;

	a=new float[length];
	b=new float[length];
	c=new float[length];
	d=new float[length];
	a_p=new float[length];
	b_p=new float[length];
	c_p=new float[length];
	d_p=new float[length];
	a_n=new float[length];
	b_n=new float[length];
	c_n=new float[length];
	d_n=new float[length];
	delta_phi=new float[length];



	for(int p=0;p<length;p++)
	{	
		if (p>=height)
		{
			a[p]=phi[p]-phi[p-height];
			if (a[p]>=0)
			{	a_p[p]=a[p];
				a_n[p]=0;
			}
			else 
			{
				a_p[p]=0;
				a_n[p]=a[p];
			}
		}
		else 
		{	
			a[p]=0;
			a_p[p]=0;
			a_n[p]=0;
		}


	}

	for(int p=0;p<length;p++)
	{	
		if (p<length-height)
		{
			b[p]=phi[p+height]-phi[p];
			if (b[p]>=0)
			{	b_p[p]=b[p];
				b_n[p]=0;
			}
			else 
			{
				b_p[p]=0;
				b_n[p]=b[p];
			}
		}
		else 
		{	
			b[p]=0;
			b_p[p]=0;
			b_n[p]=0;
		}


	}


	for(int p=0;p<length;p++)
	{	
		if (p%height!=0)
		{
			c[p]=-phi[p]+phi[p-1];
			if (c[p]>=0)
			{	c_p[p]=c[p];
				c_n[p]=0;
			}
			else 
			{
				c_p[p]=0;
				c_n[p]=c[p];
			}
		}
		else 
		{	
			c[p]=0;
			c_p[p]=0;
			c_n[p]=0;
		}


	}


	for(int p=0;p<length;p++)
	{	
		if (p%height!=(height-1))
		{
			d[p]=-phi[p+1]+phi[p];
			if (d[p]>=0)
			{	d_p[p]=d[p];
				d_n[p]=0;
			}
			else 
			{
				d_p[p]=0;
				d_n[p]=d[p];
			}
		}
		else 
		{	
			d[p]=0;
			d_p[p]=0;
			d_n[p]=0;
		}


	}


		for(int p=0;p<length;p++)
	{	
		if (phi[p]>0)
		{	
			val=a_p[p]*a_p[p];
			if (b_n[p]*b_n[p]>val) val=b_n[p]*b_n[p];
			if (c_n[p]*c_n[p]>val) val=c_n[p]*c_n[p];
			if (d_p[p]*d_p[p]>val) val=d_p[p]*d_p[p];
			delta_phi[p]=sqrt(val)-1;
		}
		else if (phi[p]<0) 
		{	
			val=a_n[p]*a_n[p];
			if (b_p[p]*b_p[p]>val) val=b_p[p]*b_p[p];
			if (c_p[p]*c_p[p]>val) val=c_p[p]*c_p[p];
			if (d_n[p]*d_n[p]>val) val=d_n[p]*d_n[p];
			delta_phi[p]=sqrt(val)-1;
		}
		else if (phi[p]==0)
			delta_phi[p]=0;



		phi[p]=phi[p]-dt*(phi[p]/sqrt(phi[p]*phi[p]+1))*delta_phi[p];
		

	}

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] d;
	delete[] a_p;
	delete[] b_p;
	delete[] c_p;
	delete[] d_p;
	delete[] a_n;
	delete[] b_n;
	delete[] c_n;
	delete[] d_n;
	delete[] delta_phi;
	return 1;


}

float MaxSeq(float a,float b,float c,float d)
{   float val;
	val=a;
	if (b>val) val=b;
	if (c>val) val=c;
	if (d>val) val=d;
	return val;

}


inline int ComputeKappa(float* phi,float* kappa ,char* mask_b, int length,int height)
{
	float Dx;
	float Dy;
	float Dxx;
	float Dyy;
	float Dxy;
	float DxDx;
	float DyDy;
	int Xp=1;
	int Xm=1;
	int Yp=height;
	int Ym=height;
	int Xden=2;
	int Yden=2;
	memset(kappa,0,length*sizeof(float));

	for (int p=0;p<length;p++)
	{
		if (mask_b[p]==1)
		{	Dx=Dxx=Dy=Dyy=0;
			Xp=Xm=1; Yp=Ym=height; Xden=Yden=2;
			if (p<height) {Ym=0; Yden=1;}
			if (p>=length-height) {Yp=0; Yden=1;}
			if (p%height==0) {Xm=0; Xden=1;}
			if (p%height==height-1) {Xp=0; Xden=1;}

	
			Dx=(phi[p+Xp]-phi[p-Xm])/2; DxDx=Dx*Dx;                    
			Dy=(phi[p+Yp]-phi[p-Ym])/2; DyDy=Dy*Dy;
			if (Xp>0 && Xm>0) Dxx=phi[p+Xp]-2*phi[p]+phi[p-Xm];                          
			if (Yp>0 && Ym>0) Dyy=phi[p+Yp]-2*phi[p]+phi[p-Ym];                          
			Dxy=(phi[p+Xp+Yp]+phi[p-Xm-Ym]-phi[p+Xp-Ym]-phi[p-Xm+Yp])*2/Xden/Yden;     
			if (kappa[p]=DxDx+DyDy)kappa[p]=(DxDx*Dyy-Dx*Dy*Dxy+DyDy*Dxx)/kappa[p];               
		}
		
	}
	return 1;
}

enum {XPOS=1, XNEG=2, YPOS=4, YNEG=8};
int ComputeGradient(float* image ,char* mask,int* edgebits,float* gx,float* gy, int length,int height)
{
	
	

	float imagepx;
	float imagemx;
	float imagepy;
	float imagemy;
	memset(gx,0,length*sizeof(float));
	memset(gy,0,length*sizeof(float));



	for (int p=1;p<length;p++)
	{
		if(mask[p]==1)
		{
			imagepx=!(edgebits[p]&XPOS) ? image[p+1] : image[p];
			imagemx=!(edgebits[p]&XNEG) ? image[p-1] : image[p];
			imagepy=!(edgebits[p]&YPOS) ? image[p+height] : image[p];
			imagemy=!(edgebits[p]&YNEG) ? image[p-height] : image[p];
			gx[p]=(imagepy-imagemy)/2;
			gy[p]=-(imagepx-imagemx)/2; //pixels are counted form top to bottom
		}
			
	}

	return 1;
}

int PlotDescriptor(vector<float*> d,vector<float*> d2,int height, int width)
{
	int lengthIndex;
	int widthIndex;
	int imIndex;
	int dSize=d.size();
	int Ibiglength=int(sqrt((double)dSize))+1;
	int Ibigwidth=Ibiglength;

	Mat Ibig(Size((int)Ibiglength*height/2, (int)Ibigwidth*width/2), CV_8UC1);
	
	for (int row=0;row<(int)Ibiglength*height/2;row++)
	{
		for (int col=0;col<(int)Ibigwidth*width/2;col++)
		{	
			if (col==127)
				printf("");
			Ibig.at<uchar>(row,col)=(uchar)0;
			lengthIndex=int((2*row)/height);
			widthIndex=int((2*col)/width);
			if (lengthIndex*Ibiglength+widthIndex<dSize)
			{	imIndex=(2*row)%256+((2*col)%256)*256;
			if (imIndex>256*256-1) 
			{printf("invalid pixel index");
			printf("");}
				Ibig.at<uchar>(row,col)=(uchar)(0.2*d[lengthIndex*Ibiglength+widthIndex][imIndex])+(uchar)(0.2*d2[lengthIndex*Ibiglength+widthIndex][imIndex]);
			}
		}
	}

	imshow("Descriptor_window", Ibig);
	waitKey(1);

	return 1;

}