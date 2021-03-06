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

#ifndef	RNDFAST_CUH
#define RNDFAST_CUH

#include "seedgen.h"
#include "..\common.h"

namespace	cudahelp
{

namespace	rand
{

__device__	uint4*	devUint4State = 0;
uint4*		hostUint4State = 0;

__global__	void	InitDeviceState(uint4*	hstate)
{
	devUint4State = hstate;
}

__host__	bool	InitFastRngs(int size)
{
	hostUint4State = GenerateRandomSeedsGPUUint4(size);

	InitDeviceState<<<1,1>>>(hostUint4State);

	cudaThreadSynchronize();

	return	CheckCUDAError("InitFastRngs");
}

__host__	void	DeinitFastRngs()
{
	if(hostUint4State)
		cudaFree(hostUint4State);
}

__device__	inline	uint4&	GetFastRngState(unsigned int i)
{
	return	devUint4State[i];
}

__device__	inline	void	SaveFastRngState(unsigned int i, uint4& state)
{
	devUint4State[i] = state;
}

__device__ inline unsigned int	LCGRng(unsigned int& seed)
{
	seed = (seed * 1664525 + 1013904223UL);
	return seed;
}

__device__ inline unsigned int	TausRng(unsigned int& z, int S1, int S2, int S3, unsigned long M)
{
	unsigned int	b = (((z << S1) ^ z) >> S2);

	z = (((z & M) << S3) ^ b);

	return  z;
}

__device__ float	HybridTausRng(uint4&	state)
{
	return 2.3283064365387e-10 * (
			TausRng(state.x, 13, 19, 12, 4294967294UL) ^
			TausRng(state.y, 2, 25, 4, 4294967288UL) ^
			TausRng(state.z, 3, 11, 17, 4294967280UL) ^
			LCGRng(state.w)
	);
}

__device__ unsigned int	HybridTausRngInt(uint4&	state)
{
	return (
			TausRng(state.x, 13, 19, 12, 4294967294UL) ^
			TausRng(state.y, 2, 25, 4, 4294967288UL) ^
			TausRng(state.z, 3, 11, 17, 4294967280UL) ^
			LCGRng(state.w)
	);
}

__device__	inline unsigned int	RngXor128(uint4&	s)
{
	unsigned int t;

	t = s.x^(s.x<<11);

	s.x = s.y;
	s.y = s.z;
	s.z = s.w;

	s.w = (s.w^(s.w>>19))^(t^(t>>8));

	return s.w;
}

struct	TaurusWrapper
{
	__device__	inline unsigned int	operator()(uint4& state)
	{
		return HybridTausRngInt(state);
	}
};

struct	Xor128Wrapper
{
	__device__	inline unsigned int	operator()(uint4& state)
	{
		return RngXor128(state);
	}
};

template	<class	GeneratorInt>
class	FastUint4Rng
{
public:
	__device__	FastUint4Rng(int i): state(devUint4State[i]), idx(i) {}
	__device__	~FastUint4Rng()
	{
		devUint4State[idx] = state;
	}

	__device__	inline unsigned int		GetInt()
	{
		return gen(state);
	}

	__device__	inline float			GetFloat()
	{
		return	2.3283064365387e-10*gen(state);
	}

	__device__	inline unsigned int		GetInt(unsigned int a, unsigned int b)
	{
		return a + gen(state)%(b-a);
	}

	__device__	inline unsigned int		GetInt(unsigned int m)
	{
		return gen(state)%m;
	}

private:
	GeneratorInt	gen;
	uint4	state;
	int		idx;
};

typedef	FastUint4Rng<TaurusWrapper>	TausGenerator;
typedef	FastUint4Rng<Xor128Wrapper>	Xor128Generator;

}
}

#endif
