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

#ifndef DEVMANAGER_H_
#define DEVMANAGER_H_

#ifdef	_MSC_VER
#define BOOST_USE_WINDOWS_H
#endif

#include <boost/shared_ptr.hpp>

#include <string>

#include <cuda_runtime.h>

namespace	cudahelp
{
	class	DeviceManager
	{
	public:

		class	Device 
		{
		public:
			Device(int	id, int total);
			~Device();

			boost::shared_ptr<Device>	GetNext() const;
			bool						HasNext() const;

			std::string	GetName() const;
			
			bool	SupportMapHost() const;
			bool	SupportOverlaps() const;

			void	EnableMapHost();

			void	SetCurrent();

			void	Print(std::ostream& out) const;

			const	cudaDeviceProp&	GetProps() const { return deviceProp_; }

		private:
			int	id_;
			int	total_;
			cudaDeviceProp deviceProp_;
		};

		typedef	boost::shared_ptr<Device>	DevicePtr;

		static	DeviceManager&	Get();
		static	void	Destroy();

		DevicePtr	GetFirstDevice() const;
		DevicePtr	GetMaxGFpsDevice() const;
		DevicePtr	GetCurrentDevice() const;

		int			GetDeviceCount() const;

	private:

		static	DeviceManager*	instance_;

		int		totalDevices_;

		DeviceManager();
		~DeviceManager();

	};
}


#endif /* DEVMANAGER_H_ */
