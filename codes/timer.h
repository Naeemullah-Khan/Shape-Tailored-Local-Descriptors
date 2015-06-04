
#pragma once

#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <time.h>
#include <windows.h>
#include <stdio.h>
#include <assert.h>

/*! Timer class for timing purposes */
class timer
{
    LARGE_INTEGER t;	
public:
	/*! Start timer */
	void tic();

	/*! Stop timer and return elapsed time in milliseconds */
	double toc();
};

