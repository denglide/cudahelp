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

#include "kernels.h"

__global__	void	MultiplyInplaceDev(int* data, int size)
{
	const int	tid = blockDim.x*blockIdx.x + threadIdx.x;
	if(tid < size)
		data[tid] *= 2;
}

__global__	void	MultiplyDev(int* in_data, int* out_data, int size)
{
	const int	tid = blockDim.x*blockIdx.x + threadIdx.x;
	if(tid < size)
	{
		int	data = in_data[tid];
		out_data[tid] = data*2;
	}
}

void	MultiplyInplaceGPU(int* data, int size)
{
	const int	numThreads = 512;
	const int	numBlocks = (size + numThreads - 1)/numThreads;

	MultiplyInplaceDev<<<numBlocks,numThreads>>>(data, size);
}

void	MultiplyGPU(int* in_data, int* out_data, int size)
{
	const int	numThreads = 512;
	const int	numBlocks = (size + numThreads - 1)/numThreads;

	MultiplyDev<<<numBlocks,numThreads>>>(in_data, out_data, size);
}

void	MultiplyGPUStreams(int* in_data, int* out_data, int size, cudaStream_t s)
{
	const int	numThreads = 512;
	const int	numBlocks = (size + numThreads - 1)/numThreads;

	MultiplyDev<<<numBlocks,numThreads,0,s>>>(in_data, out_data, size);
}

