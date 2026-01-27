#pragma once

#include "../../include/flowVk/Instance.hpp"
#include <vulkan/vulkan.h>
#include "../../include/external/vk_mem_alloc.h"

#include <unordered_map>
#include <functional>
#include <set>
namespace Flow {


struct InstanceImpl {
	VkInstance 		 instance = VK_NULL_HANDLE;
	VkPhysicalDevice physical = VK_NULL_HANDLE;
	VkDevice 		 device   = VK_NULL_HANDLE;

	uint32_t computeQueueFamily = UINT32_MAX;
	VkQueue  computeQueue       = VK_NULL_HANDLE;

	VmaAllocator allocator = VK_NULL_HANDLE;

	VkCommandPool cmdPool = VK_NULL_HANDLE;
	
	struct KernelState {
		VkShaderModule shaderModule = VK_NULL_HANDLE;
		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
		VkPipeline pipeline = VK_NULL_HANDLE;
		std::vector<VkDescriptorSetLayout> setLayouts;
	};

	struct BufferState {
		std::string name;
		BufferAccess access = BufferAccess::ReadOnly;

		VkBuffer buffer = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		std::size_t sizeBytes = 0;
	};

	std::unordered_map<std::string, KernelState> kernels;
	std::unordered_map<std::string, BufferState> buffers;

	~InstanceImpl();
	void submit_one_time(std::function<void(VkCommandBuffer)> record);

};

} //namespace Flow