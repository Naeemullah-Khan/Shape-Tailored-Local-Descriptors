#include "Params.h"

int ComputeDirectionalGradient(float** image,int** mask,int width,int height,int bins_num,float** gx,float** gy,float*** im_grad_matrix,int** bin_matrix)
{

	for(int i=1;i<height-1;i++)
	{
		for(int j=1;j<width-1;j++)
		{
			gx[i][j]=0;
			gy[i][j]=0;
		}
	}

	//mask is ONE (check/checked) I_big is calculated on the entire image instead of the mask only
	for(int i=1;i<height-1;i++)
	{
		for(int j=1;j<width-1;j++)
		{
			mask[i][j]=1;
		}
	}





	for(int i=1;i<height-1;i++)
	{
		for(int j=1;j<width-1;j++)
		{
			if(mask[i][j]>0) 
			{
				if(mask[i+1][j]==0) image[i+1][j]=image[i][j];
				if(mask[i-1][j]==0) image[i-1][j]=image[i][j];
				if(mask[i][j+1]==0) image[i][j+1]=image[i][j];
				if(mask[i][j-1]==0) image[i][j-1]=image[i][j];				
				gy[i][j]= (image[i+1][j]-image[i-1][j])/2 ;
				gx[i][j]= (image[i][j+1]-image[i][j-1])/2 ;
			}
		}
	}

	for(int i=0;i<width;i++) gx[0][i]=gx[1][i];
	for(int i=0;i<width;i++) gx[height-1][i]=gx[height-2][i];
	for(int i=0;i<height;i++) gx[i][0]=gx[i][1];
	for(int i=0;i<height;i++) gx[i][width-1]=gx[i][width-2];

	for(int i=0;i<width;i++) gy[0][i]=gx[1][i];
	for(int i=0;i<width;i++) gy[height-1][i]=gx[height-2][i];
	for(int i=0;i<height;i++) gy[i][0]=gx[i][1];
	for(int i=0;i<height;i++) gy[i][width-1]=gx[i][width-2];

	for(int i=1;i<height-1;i++)
	{
		for(int j=1;j<width-1;j++)
		{
			if(gy[i][j]!=0) gy[i][j]=gy[i][j]*(-1);
		}
	}

	for(int k=1;k<=step_theeta;k++)
	{
		for(int i=0;i<height;i++)
		{
			for(int j=0;j<width;j++)
			{
				im_grad_matrix[i][j][k-1]=(float)abs( (gx[i][j]*cos((k*pi)/step_theeta))+(gy[i][j]*sin((k*pi)/step_theeta)) );
			}
		}
	}
	
	int start_bin=0;
	for(int i=0;i<bins_theeta;i++)
	{
		start_bin=(int) (step_theeta/bins_theeta)*(i);
		for(int j=0;j<bins_num;j++)
		{	
			if((start_bin+j)>step_theeta-1)	bin_matrix[i][j]= (int) ((start_bin+j)/step_theeta);
			else bin_matrix[i][j]= (start_bin+j);
		}
	}

	return 1;
}
