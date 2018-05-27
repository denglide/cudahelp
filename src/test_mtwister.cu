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

#include "random/mtwister.cuh"
#include "kernels.h"
#include "common.h"

__global__	void	TestMersenneTwisterGPU(int*	data, int count)
{
	const	int		tid = blockDim.x*blockIdx.x + threadIdx.x;
	const	int	numThreads = blockDim.x*gridDim.x;

	cudahelp::rand::mt_state_t	state = cudahelp::rand::GetMTState(tid);
	cudahelp::rand::mt_struct_stripped_t	config = cudahelp::rand::GetMTConfig(tid);

	for(int idx = tid; idx < count; idx += numThreads)
		data[idx] = cudahelp::rand::MTwisterRndInt(state, config);

	cudahelp::rand::SaveMTState(tid, state);
}       

__global__	void	TestMersenneTwisterClassGPU(int*	data, int count)
{
	int		idx = blockDim.x*blockIdx.x + threadIdx.x;
	const	int	numThreads = blockDim.x*gridDim.x;

	cudahelp::rand::MTGenerator	gen(idx);

	for(; idx < count; idx += numThreads)
		data[idx] = gen.GetInt();
}

void	TestMersenneTwister(int*	data, int count, int rngs)
{
	int	numThreads = 256;
	int	numBlocks = cudahelp::GetNumberOfBlocks(numThreads, rngs);

	TestMersenneTwisterGPU<<<numBlocks, numThreads>>>(data, count);

	cudaThreadSynchronize();

	cudahelp::CheckCUDAError("TestMersenneTwisterGPU");

}

void	TestMersenneTwisterClass(int*	data, int count, int rngs)
{
	int	numThreads = 256;
	int	numBlocks = cudahelp::GetNumberOfBlocks(numThreads, rngs);

	TestMersenneTwisterClassGPU<<<numBlocks, numThreads>>>(data, count);

	cudaThreadSynchronize();

	cudahelp::CheckCUDAError("TestMersenneTwisterClassGPU");
}
