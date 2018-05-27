/*
* MIT License
*
* Copyright(c) 2010 Denis Gladkov
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files(the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions :
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include "common.h"
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>

namespace	cudahelp
{

bool CheckCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();

    if( cudaSuccess != err)
	{
		std::cerr<<"Cuda error in "<<msg<<" "<<cudaGetErrorString(err)<<"\n";
		return false;
	}

	return true;
}

int		GetNumberOfBlocks(int	numThreads, int	numSamples)
{
	int	numBlocks = numSamples/numThreads;

	if(numSamples%numThreads)
		numBlocks++;

	return	numBlocks;
}

void	CheckCUDAErrorAndThrow(const char*	msg)
{
    cudaError_t err = cudaGetLastError();

    if( cudaSuccess != err)
	{
		std::ostringstream	out;
		out<<"Cuda error in "<<msg<<" "<<cudaGetErrorString(err);

		throw	std::runtime_error(out.str());
	}
}

}