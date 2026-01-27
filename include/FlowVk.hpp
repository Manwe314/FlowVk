#pragma once

#include "flowVk/ShaderMeta.hpp"
#include "flowVk/Instance.hpp"
#include "flowVk/Buffer.hpp"


#if defined(__has_include)
	#if __has_include("KernelBuffers.hpp")
		#include "KernelBuffers.hpp"
		#define FLOWVK_HAS_KERNEL_BUFFERS 1
	#else
		#define FLOWVK_HAS_KERNEL_BUFFERS 0
	#endif
#else
	#define FLOWVK_HAS_KERNEL_BUFFERS 0
#endif
