#include <math.h>
#include <stdio.h>
#include <iostream>

using namespace std;

/*
 * Solves the system Ax = b using the conjugate gradient method.  
 * Assumes A is linear, symmetric, positive definite.
 *
 * The matrix A is not stored, instead the class uses a function pointer
 * such that when the function pointed to is called, Ax is computed.
 */

template <class T> class ConjugateGradient
{
 public:
  int N; //size of matrix

  T *x, *b, *r, *p, *Ap;
  
  char *active;
  int *edgebits;
  char *f;
  
  float alpha;
  int XSize,YSize;
  
 public:
  ConjugateGradient() {
    N=0;
    r=p=Ap=0;
  };

  int allocate(int Size,int width, int height);
  void deallocate();

  void computeSolution(float errorTol);
  void initialize();

 private:
  // c = scalea * a + scaleb * b
  void  add(T *a, float scalea, T *b, float scaleb, T*c);
  // a := b
  void  copy(T *a, const T* b);
  // sum_i a[i]*b[i]
  float inner(const T* a, const T* b);
  float error();
  void print(T *x);
  int ComputeAx(T *_x,T *_Ax);
};


template <class T>
int ConjugateGradient<T>::allocate(int Size,int width, int height) 
{
  if ( Size <= 0 ) return 0;
  N=Size;
  XSize=width;
  YSize=height;

  if ( !(r  = new T[N]) ||
       !(p  = new T[N]) ||
       !(Ap = new T[N])  ) {
    deallocate();
    return 0;
  }

  return 1;
}


template <class T>
void ConjugateGradient<T>::deallocate()
{
  delete[]  r;  r=0;
  delete[]  p;  p=0;
  delete[] Ap; Ap=0;

  return;
}

template <class T>
int ConjugateGradient<T>::ComputeAx(T *_x, T *_Ax)
{
	enum {XPOS=1, XNEG=2, YPOS=4, YNEG=8};
	T xpx,xmx,xpy,xmy;
	for (int p=0; p<N; p++) 
	{
		if(active[p]==1)
		{
			xpx=!(edgebits[p]&XPOS) ? _x[p+1] : _x[p-1];
			xmx=!(edgebits[p]&XNEG) ? _x[p-1] : _x[p+1];
			xpy=!(edgebits[p]&YPOS) ? _x[p+XSize] : _x[p-XSize];
			xmy=!(edgebits[p]&YNEG) ? _x[p-XSize] : _x[p+XSize];
			_Ax[p]=_x[p] - (xpx+xmx+xpy+xmy-_x[p]*4)*alpha;
		}
	}

  return 1;
}

template <class T>
void ConjugateGradient<T>::computeSolution(float errorTol)
{
  float alpha, beta;
  float err;
  T *Ax=r;
  float tol=1;

  float bb=inner(b,b);
  tol=bb>0 ? errorTol*errorTol*bb : tol;
  
  //initialize();                  // x = 0
  int numbActive=0;
  for (int i=0; i<N; i++) if (active[i]&&numbActive<100) numbActive++; //remove the numbActive==100
  
  ComputeAx(x,Ax);                      // Ax = A(x)
  add(Ax, -1, b, 1, r);          // r = b - Ax
  copy(p, r);                    // p = r
  float rr=inner(r,r), rrnew;
  int count=0;

  //printf("rr start=%f\n", rr);
  
  while (rr > tol && count < 2*numbActive) {

    ComputeAx(p, Ap);                    // Ap = A(p)
    alpha=rr/inner(p,Ap);        // alpha = r.*r / p .* Ap
    //printf("iter.=%d/%d, err=%f, errorTol=%f, alpha=%f\n", count, numbActive, rr, tol, alpha);
    add( x, 1,  p,  alpha, x);   // x <= x + alpha *  p
    add( r, 1, Ap, -alpha, r);   // r <= r - alpha * Ap

    //err=error();

    rrnew=inner(r,r);
    beta=rrnew/rr;
    add( r, 1, p, beta, p);      // p <= r + beta*p
    rr=rrnew;

    count++;
  }

  //printf("iter.=%d/%d, err=%f, errorTol=%f\n", count, numbActive, rr, tol);
  cout<<count<<endl;
  return;

}


template <class T>
void ConjugateGradient<T>::add(T *a, float scalea, T *b, float scaleb, T*c)
{
  for (int i=0; i<N; i++)
    if (active[i]) c[i]=a[i]*scalea + b[i]*scaleb;

  return;
}


template <class T>
void ConjugateGradient<T>::copy(T *a, const T* b)
{
  for (int i=0; i<N; i++)
    if (active[i]) a[i]=b[i];

  return;
}


template <class T>
float ConjugateGradient<T>::inner(const T* a, const T* b)
{
  float ret=0;
 
  for (int i=0; i<N; i++) {
	   if (active[i]) ret+=a[i]*b[i];
  }
  
  return ret;
}


template <class T>
void ConjugateGradient<T>::initialize()
{
  for (int i=0; i<N; i++) if (!active[i]) x[i]=0;

  return;
}


template <class T>
float ConjugateGradient<T>::error()
{
  float maxerr=0;
  float err;
  
  for (int i=0; i<N; i++) {
    if (active[i]) {
      err=E(r[i]);
      maxerr=err > maxerr ? err : maxerr;
    }
  }

  return maxerr;
}

template <class T>
void ConjugateGradient<T>::print(T* x)
{
  printf("(");
  for (int i=0; i<N; i++) {
    if (active[i]) {
      if ( i<N-1 )
	printf("%5.2f, ", x[i]);
      else 
	printf("%5.2f )\n", x[i]);
    }
  }

  return;
}

