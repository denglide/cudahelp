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

#include "test_rngs.h"
#include "random/mtwister.h"
#include "random/rndfast.h"
#include "devmanager.h"
#include "benchmark.h"
#include "kernels.h"
#include <iostream>

using namespace std;

using namespace	cudahelp;

void	test_rngs_gpu()
{
	std::cout<<"Testing Mersenne Twister RNG\n";

	DeviceManager::DevicePtr	dev = DeviceManager::Get().GetMaxGFpsDevice();
	dev->SetCurrent();

	const int	rngsCount = 65536;
	const	int	numSamples = 0x10000000>>3;

	int*	data = 0;

	if(!rand::InitGPUTwisters("data/MersenneTwister.dat",rngsCount,999))
	{
		std::cout<<"Can't init twisters\n";
		return;
	}

	cudaMalloc(&data, numSamples*sizeof(int));

	std::cout<<"Testing	"<<numSamples<<" samples\n";

	CudaBenchmark	bench;

	bench.Start("RandGen");

	TestMersenneTwisterClass(data, numSamples, rngsCount);

	bench.Stop("RandGen");

	std::cout<<numSamples<<" generated in "<<bench.GetValue("RandGen")<<" ms\n";

	cudaFree(data);

	rand::DeinitGPUTwisters();
	
	cudaThreadExit();
}

void	test_fast_rngs_gpu()
{
	std::cout<<"Testing Fast RNGs\n";

	DeviceManager::DevicePtr	dev = DeviceManager::Get().GetMaxGFpsDevice();
	dev->SetCurrent();

	const int	rngsCount = 65536;
	const	int	numSamples = 0x10000000>>3;

	int*	data = 0;

	if(!rand::InitFastRngs(rngsCount))
	{
		std::cout<<"Can't init RNGs\n";
		return;
	}

	cudaMalloc(&data, numSamples*sizeof(int));

	std::cout<<"Testing	"<<numSamples<<" samples\n";

	CudaBenchmark	bench;

	bench.Start("FastRandGen");

	TestFastRngs(data, numSamples, rngsCount);

	bench.Stop("FastRandGen");

	std::cout<<numSamples<<" generated in "<<bench.GetValue("FastRandGen")<<" ms\n";

	cudaFree(data);

	rand::DeinitFastRngs();

	cudaThreadExit();
}