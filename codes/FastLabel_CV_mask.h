#include <vector>

int Initialize(std::vector<float*> I_big_vector_all,int* alpha_vector,int width, int height,int feature_vector_size,int** lables_init,int nLabel,double dSigma,int nIter,int nPlot,double *pL_out, double *pRecon_out, double *pErr_out);
int FastLabel_CV_mask(double* pI, int* pL, int* pM, double *pL_out, double *pRecon_out, double *pErr_out, int nRow, int nCol, int nDep, int nVec, int nLabel, int nIter, int nPlot, double dSigma);
