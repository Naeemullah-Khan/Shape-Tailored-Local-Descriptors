#include "timer.h"

// Timer class for timing purposes
void timer::tic()	
{
	QueryPerformanceCounter(&t);
}

double timer::toc()
{
	LARGE_INTEGER frequency;
	LARGE_INTEGER t1;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&t1);
	double	elapsedTime = (t1.QuadPart - t.QuadPart) * 1000.0 / frequency.QuadPart;
	return elapsedTime;
}