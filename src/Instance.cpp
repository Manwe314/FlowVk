#include "../include/flowVk/Instance.hpp"
#include "internal/InstanceImpl.hpp" 

#include <stdexcept>
#include <iostream>
#include <cstring>
#include <fstream>
#include <set>

#include <vulkan/vulkan.h>

#define VMA_IMPLEMENTATION
#include "../include/external/vk_mem_alloc.h"

#ifdef FLOWVK_WITH_KERNEL_REGISTRY
  #include "KernelBuffers.hpp"
#endif

namespace Flow {

// ----- Helpers -----

static void vkCheck(VkResult result, const char* msg)
{
	if (result != VK_SUCCESS)
		throw std::runtime_error(std::string("FlowVk Vulkan error: ") + msg + " (VkResult=" + std::to_string((int)result) + ")");
}

static std::vector<uint32_t> read_spirv_words(const std::filesystem::path& path)
{
	std::ifstream file(path, std::ios::binary);
	if (!file)
		throw std::runtime_error("FlowVk: failed to open SPV: " + path.string());

	file.seekg(0, std::ios::end);
	std::streamoff size = file.tellg();
	if (size <= 0)
		throw std::runtime_error("FlowVk: SPV file is empty: " + path.string());
	if ((size % 4) != 0)
		throw std::runtime_error("FlowVk: SPV size not multiple of 4: " + path.string());

	std::vector<uint32_t> words(static_cast<size_t>(size / 4));
	file.seekg(0, std::ios::beg);
	file.read(reinterpret_cast<char*>(words.data()), size);
	if (!file)
		throw std::runtime_error("FlowVk: failed to read SPV: " + path.string());
	return words;
}

static VkBufferUsageFlags ssbo_usage()
{
	return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
}

static void create_vma_buffer(InstanceImpl* pimpl, VkDeviceSize size, VkBuffer* outBuf, VmaAllocation* outAlloc)
{
	VkBufferCreateInfo bufferCreateInfo{};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.size = size;
	bufferCreateInfo.usage = ssbo_usage();
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo allocationCreateInfo{};
	allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
	allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
	        					 VMA_ALLOCATION_CREATE_MAPPED_BIT;

	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation alloc = VK_NULL_HANDLE;
	VkResult r = vmaCreateBuffer(pimpl->allocator, &bufferCreateInfo, &allocationCreateInfo, &buffer, &alloc, nullptr);
	vkCheck(r, "vmaCreateBuffer");
	*outBuf = buffer;
	*outAlloc = alloc;
}

static void zero_fill_buffer(InstanceImpl* pimpl, VkBuffer buffer, VkDeviceSize sizeBytes) {
	pimpl->submit_one_time([&](VkCommandBuffer cmd) {
		vkCmdFillBuffer(cmd, buffer, 0, sizeBytes, 0);

		VkBufferMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer = buffer;
		barrier.offset = 0;
		barrier.size = sizeBytes;

		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0,
			0, nullptr,
			1, &barrier,
			0, nullptr
		);
	});
}

static std::vector<const char*> default_instance_extensions(bool /*validation*/)
{
	return {};
}

static std::vector<const char*> default_device_extensions()
{
	return {};
}

static uint32_t find_compute_queue_family(VkPhysicalDevice physicalDevice)
{
	uint32_t count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
	std::vector<VkQueueFamilyProperties> props(count);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, props.data());

	for (uint32_t i = 0; i < count; ++i)
	{
		if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
			return i;
  	}
	return UINT32_MAX;
}

static bool device_name_contains(VkPhysicalDevice phisicalDevice, const std::string& needle)
{
	if (needle.empty())
		return true;
	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(phisicalDevice, &properties);
	return std::string(properties.deviceName).find(needle) != std::string::npos;
}

static VkPhysicalDevice pick_physical_device(VkInstance instance, const InstanceConfig& config, uint32_t& outComputeQF)
{
	uint32_t count = 0;
	vkCheck(vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices(count)");
	if (count == 0)
		throw std::runtime_error("FlowVk: No Vulkan physical devices found");

	std::vector<VkPhysicalDevice> devices(count);
	vkCheck(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "vkEnumeratePhysicalDevices(list)");

	for (auto phisicalDevice : devices)
	{
		if (!device_name_contains(phisicalDevice, config.prefer_device_name_contains))
			continue;
		uint32_t queueFamily = find_compute_queue_family(phisicalDevice);
		if (queueFamily != UINT32_MAX)
		{
			outComputeQF = queueFamily;
			return phisicalDevice;
		}
	}

	for (auto phisicalDevice : devices)
	{
    	uint32_t queueFamily = find_compute_queue_family(phisicalDevice);
    	if (queueFamily != UINT32_MAX)
		{
      		outComputeQF = queueFamily;
      		return phisicalDevice;
    	}
	}

	throw std::runtime_error("FlowVk: No Vulkan device with a compute queue was found");
}


InstanceImpl::~InstanceImpl()
{
	for (auto& [name, kernel] : kernels)
	{
		for (auto layout : kernel.setLayouts)
			if (layout)
				vkDestroyDescriptorSetLayout(device, layout, nullptr);
		
		if (kernel.pipeline)		vkDestroyPipeline(device, kernel.pipeline, nullptr);
		if (kernel.pipelineLayout)	vkDestroyPipelineLayout(device, kernel.pipelineLayout, nullptr);
		if (kernel.shaderModule)	vkDestroyShaderModule(device, kernel.shaderModule, nullptr);
	}
	kernels.clear();
	
	for (auto& [n, b] : buffers)
	{
		if (b.buffer)
			vmaDestroyBuffer(allocator, b.buffer, b.allocation);
	}
	buffers.clear();

	if (cmdPool)	vkDestroyCommandPool(device, cmdPool, nullptr);
	if (allocator)	vmaDestroyAllocator(allocator);
	if (device)		vkDestroyDevice(device, nullptr);
	if (instance)	vkDestroyInstance(instance, nullptr);
}

void InstanceImpl::submit_one_time(std::function<void(VkCommandBuffer)> record)
{
	VkCommandBufferAllocateInfo allocateInfo{};
	allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocateInfo.commandPool = cmdPool;
	allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocateInfo.commandBufferCount = 1;

	VkCommandBuffer cmd = VK_NULL_HANDLE;
	vkCheck(vkAllocateCommandBuffers(device, &allocateInfo, &cmd), "vkAllocateCommandBuffers");

	VkCommandBufferBeginInfo bufferBeginInfo{};
	bufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkCheck(vkBeginCommandBuffer(cmd, &bufferBeginInfo), "vkBeginCommandBuffer");

	record(cmd);

	vkCheck(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

	VkSubmitInfo subbmitInfo{};
	subbmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	subbmitInfo.commandBufferCount = 1;
	subbmitInfo.pCommandBuffers = &cmd;

	VkFenceCreateInfo fenceCreateInfo{};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	VkFence fence = VK_NULL_HANDLE;
	vkCheck(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence), "vkCreateFence");

	vkCheck(vkQueueSubmit(computeQueue, 1, &subbmitInfo, fence), "vkQueueSubmit");
	vkCheck(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");

	vkDestroyFence(device, fence, nullptr);
	vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
}

// ----- Public Api -----

Instance makeInstance(const InstanceConfig& config)
{
	auto pimpl = std::make_shared<InstanceImpl>();

	// ----- VkInstance -----
	VkApplicationInfo app{};
	app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app.pApplicationName = "FlowVkApp";
	app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
	app.pEngineName = "FlowVk";
	app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
	app.apiVersion = VK_API_VERSION_1_3;

	auto instExtensions = config.instance_extensions;
	if (instExtensions.empty())
		instExtensions = default_instance_extensions(config.enable_validation);

	VkInstanceCreateInfo instCreateInfo{};
	instCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instCreateInfo.pApplicationInfo = &app;
	instCreateInfo.enabledExtensionCount = static_cast<uint32_t>(instExtensions.size());
	instCreateInfo.ppEnabledExtensionNames = instExtensions.empty() ? nullptr : instExtensions.data();


	vkCheck(vkCreateInstance(&instCreateInfo, nullptr, &pimpl->instance), "vkCreateInstance");

	// ----- Physical device selection -----
	pimpl->physical = pick_physical_device(pimpl->instance, config, pimpl->computeQueueFamily);

	// ----- Logical device -----
	float queuePriority = 1.0f;
	VkDeviceQueueCreateInfo queueCreateInfo{};
	queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo.queueFamilyIndex = pimpl->computeQueueFamily;
	queueCreateInfo.queueCount = 1;
	queueCreateInfo.pQueuePriorities = &queuePriority;

	auto deviceExtensions = config.device_extensions;
	if (deviceExtensions.empty())
		deviceExtensions = default_device_extensions();

  	VkPhysicalDeviceFeatures features{};

	VkDeviceCreateInfo deviceCreateInfo{};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
	deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
	deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.empty() ? nullptr : deviceExtensions.data();
	deviceCreateInfo.pEnabledFeatures = &features;

	vkCheck(vkCreateDevice(pimpl->physical, &deviceCreateInfo, nullptr, &pimpl->device), "vkCreateDevice");

	vkGetDeviceQueue(pimpl->device, pimpl->computeQueueFamily, 0, &pimpl->computeQueue);


	// ------ CMD Pool -----
	VkCommandPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = pimpl->computeQueueFamily;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	vkCheck(vkCreateCommandPool(pimpl->device, &poolInfo, nullptr, &pimpl->cmdPool), "vkCreateCommandPool");

	// ----- VMA allocator -----
	VmaAllocatorCreateInfo allocatorCreateInfo{};
	allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
	allocatorCreateInfo.physicalDevice = pimpl->physical;
	allocatorCreateInfo.device = pimpl->device;
	allocatorCreateInfo.instance = pimpl->instance;

	vkCheck(vmaCreateAllocator(&allocatorCreateInfo, &pimpl->allocator), "vmaCreateAllocator");

	Instance out;
	out.pimpl = std::move(pimpl);
	return out;
}

void Instance::addKernel(const std::string& kernelName, const std::filesystem::path& spvPath)
{
	if (!pimpl)
		throw std::runtime_error("FlowVk: addKernel called on empty Instance");

	if (pimpl->kernels.find(kernelName) != pimpl->kernels.end())
		throw std::runtime_error("FlowVk: kernel already exists: " + kernelName);

#ifndef FLOWVK_WITH_KERNEL_REGISTRY
  throw std::runtime_error(
    "FlowVk: Kernel registry not available. "
    "Did you call flowvk_add_kernels(...) for your target?"
  );
#endif

	const auto& mod = Flow::shader_meta::registry::get_module(kernelName);

	uint32_t maxSet = 0;
	for (const auto& buffer : mod.buffers)
    	maxSet = std::max(maxSet, buffer.set);
  	const uint32_t setCount = mod.buffers.empty() ? 0u : (maxSet + 1u);

	std::vector<std::vector<VkDescriptorSetLayoutBinding>> perSet(setCount);

	std::vector<std::set<uint32_t>> usedBindings(setCount);

	for (const auto& buffer : mod.buffers)
	{
		if (buffer.set >= setCount)
    		throw std::runtime_error("FlowVk: invalid set index in metadata for kernel: " + kernelName);
		if (usedBindings[buffer.set].count(buffer.binding))
    		throw std::runtime_error("FlowVk: duplicate binding in metadata for kernel: " + kernelName);

    	usedBindings[buffer.set].insert(buffer.binding);

		VkDescriptorSetLayoutBinding layoutBinding{};
		layoutBinding.binding = buffer.binding;
		layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBinding.descriptorCount = 1;
		layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		layoutBinding.pImmutableSamplers = nullptr;

		perSet[buffer.set].push_back(layoutBinding);
	}

	for (auto& vector : perSet)
		std::sort(vector.begin(), vector.end(), [](auto& a, auto& c) { return a.binding < c.binding; });

	InstanceImpl::KernelState kernel{};

	kernel.setLayouts.resize(setCount, VK_NULL_HANDLE);

	for (uint32_t set = 0; set < setCount; ++set)
	{
		VkDescriptorSetLayoutCreateInfo setLayoutCreateInfo{};
		setLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		setLayoutCreateInfo.bindingCount = static_cast<uint32_t>(perSet[set].size());
		setLayoutCreateInfo.pBindings = perSet[set].empty() ? nullptr : perSet[set].data();

		vkCheck(vkCreateDescriptorSetLayout(pimpl->device, &setLayoutCreateInfo, nullptr, &kernel.setLayouts[set]), "vkCreateDescriptorSetLayout");
	}

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
	pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCreateInfo.setLayoutCount = static_cast<uint32_t>(kernel.setLayouts.size());
	pipelineLayoutCreateInfo.pSetLayouts = kernel.setLayouts.empty() ? nullptr : kernel.setLayouts.data();
	pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
	pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

	vkCheck(vkCreatePipelineLayout(pimpl->device, &pipelineLayoutCreateInfo, nullptr, &kernel.pipelineLayout), "vkCreatePipelineLayout");

	auto words = read_spirv_words(spvPath);

	VkShaderModuleCreateInfo shaderModule{};
	shaderModule.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shaderModule.codeSize = words.size() * sizeof(uint32_t);
	shaderModule.pCode = words.data();

	vkCheck(vkCreateShaderModule(pimpl->device, &shaderModule, nullptr, &kernel.shaderModule), "vkCreateShaderModule");

	VkPipelineShaderStageCreateInfo stage{};
	stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = kernel.shaderModule;
	stage.pName = "main";
	stage.pSpecializationInfo = nullptr;

	VkComputePipelineCreateInfo pipelineCreateInfo{};
	pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineCreateInfo.stage = stage;
	pipelineCreateInfo.layout = kernel.pipelineLayout;
	pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
	pipelineCreateInfo.basePipelineIndex = -1;

	vkCheck(vkCreateComputePipelines(pimpl->device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &kernel.pipeline), "vkCreateComputePipelines");

	pimpl->kernels.emplace(kernelName, kernel);
}

void Instance::runSingleKernel(const std::string& kernelName, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
	if (!pimpl)
		throw std::runtime_error("FlowVk: runSingleKernel called on empty Instance");

	auto kernelItterator = pimpl->kernels.find(kernelName);
	if (kernelItterator == pimpl->kernels.end())
		throw std::runtime_error("FlowVk: unknown kernel: " + kernelName);

#ifndef FLOWVK_WITH_KERNEL_REGISTRY
	throw std::runtime_error(
		"FlowVk: Kernel registry not available. "
		"Did you call flowvk_add_kernels(...) for your target?"
	);
#endif

	const auto& module = Flow::shader_meta::registry::get_module(kernelName);

	uint32_t maxSet = 0;
	for (const auto& b : module.buffers)
		maxSet = std::max(maxSet, b.set);
	const uint32_t setCount = module.buffers.empty() ? 0u : (maxSet + 1u);

	auto& kernelState = kernelItterator->second;
	if (kernelState.setLayouts.size() != setCount)
		throw std::runtime_error("FlowVk: kernel setLayout count mismatch (did metadata change?): " + kernelName);

	uint32_t totalStorageBindings = static_cast<uint32_t>(module.buffers.size());

	VkDescriptorPoolSize poolSize{};
	poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSize.descriptorCount = totalStorageBindings;

	VkDescriptorPoolCreateInfo poolCreateInfo{};
	poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolCreateInfo.maxSets = setCount;
	poolCreateInfo.poolSizeCount = (totalStorageBindings > 0) ? 1u : 0u;
	poolCreateInfo.pPoolSizes = (totalStorageBindings > 0) ? &poolSize : nullptr;

	VkDescriptorPool descPool = VK_NULL_HANDLE;
	vkCheck(vkCreateDescriptorPool(pimpl->device, &poolCreateInfo, nullptr, &descPool), "vkCreateDescriptorPool");

	std::vector<VkDescriptorSet> sets(setCount, VK_NULL_HANDLE);
	if (setCount > 0)
	{
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descPool;
		allocInfo.descriptorSetCount = setCount;
		allocInfo.pSetLayouts = kernelState.setLayouts.data();	
		vkCheck(vkAllocateDescriptorSets(pimpl->device, &allocInfo, sets.data()), "vkAllocateDescriptorSets");
	}

	std::vector<VkDescriptorBufferInfo> bufferInfos;
	bufferInfos.reserve(module.buffers.size());

	std::vector<VkWriteDescriptorSet> writes;
	writes.reserve(module.buffers.size());

	for (const auto& buffer : module.buffers)
	{
		auto bufferItterator = pimpl->buffers.find(std::string(buffer.name));
		if (bufferItterator == pimpl->buffers.end())
		{
			vkDestroyDescriptorPool(pimpl->device, descPool, nullptr);
			throw std::runtime_error("FlowVk: missing required buffer '" + std::string(buffer.name) + "' for kernel '" + kernelName + "'");
		}

    	auto& state = bufferItterator->second;
    	if (!state.buffer)
		{
    	  vkDestroyDescriptorPool(pimpl->device, descPool, nullptr);
    	  throw std::runtime_error("FlowVk: buffer '" + std::string(buffer.name) + "' not allocated");
    	}


		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = state.buffer;
		bufferInfo.offset = 0;
		bufferInfo.range  = VK_WHOLE_SIZE;
		bufferInfos.push_back(bufferInfo);

		VkWriteDescriptorSet setW{};
		setW.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		setW.dstSet = sets[buffer.set];
		setW.dstBinding = buffer.binding;
		setW.dstArrayElement = 0;
		setW.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		setW.descriptorCount = 1;
		setW.pBufferInfo = &bufferInfos.back();
		writes.push_back(setW);
	}

	if (!writes.empty())
		vkUpdateDescriptorSets(pimpl->device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

	pimpl->submit_one_time([&](VkCommandBuffer cmd) {
    	if (!module.buffers.empty())
		{
      		std::vector<VkBufferMemoryBarrier> preBarriers;
      		preBarriers.reserve(module.buffers.size());

      		for (const auto& buffer : module.buffers)
			{
        		auto& state = pimpl->buffers.at(std::string(buffer.name));
        		VkBufferMemoryBarrier memBarrier{};
        		memBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        		memBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        		memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        		memBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        		memBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        		memBarrier.buffer = state.buffer;
        		memBarrier.offset = 0;
        		memBarrier.size = VK_WHOLE_SIZE;
        		preBarriers.push_back(memBarrier);
      		}

    		vkCmdPipelineBarrier(
				cmd,
				VK_PIPELINE_STAGE_HOST_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0,
				0, nullptr,
				static_cast<uint32_t>(preBarriers.size()), preBarriers.data(),
				0, nullptr
    		);
    	}

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernelState.pipeline);

		if (setCount > 0)
		{
			vkCmdBindDescriptorSets(
				cmd,
				VK_PIPELINE_BIND_POINT_COMPUTE,
				kernelState.pipelineLayout,
				0,
				setCount,
				sets.data(),
				0,
				nullptr
			);
		}

		vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ);

    	if (!module.buffers.empty())
		{
    		std::vector<VkBufferMemoryBarrier> postBarriers;
    		postBarriers.reserve(module.buffers.size());

    		for (const auto& buffer : module.buffers) {
    			auto& state = pimpl->buffers.at(std::string(buffer.name));
    			VkBufferMemoryBarrier memBarrier{};
    			memBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    			memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    			memBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    			memBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    			memBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    			memBarrier.buffer = state.buffer;
    			memBarrier.offset = 0;
    			memBarrier.size = VK_WHOLE_SIZE;
    			postBarriers.push_back(memBarrier);
    		}

			vkCmdPipelineBarrier(
				cmd,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_HOST_BIT,
				0,
				0, nullptr,
				static_cast<uint32_t>(postBarriers.size()), postBarriers.data(),
				0, nullptr
			);
    	}
	});
	vkDestroyDescriptorPool(pimpl->device, descPool, nullptr);
}

BufferBuilder Instance::makeReadOnly(const std::string& name)
{
	if (!pimpl)
		throw std::runtime_error("FlowVk: makeReadOnly on empty Instance");
	BufferBuilder buffer;
	buffer.owner = pimpl;
	buffer.name = name;
	buffer.access = BufferAccess::ReadOnly;
	buffer.zero_initialize = false;
	return buffer;
}

BufferBuilder Instance::makeWriteOnly(const std::string& name)
{
	if (!pimpl)
		throw std::runtime_error("FlowVk: makeWriteOnly on empty Instance");
	BufferBuilder buffer;
	buffer.owner = pimpl;
	buffer.name = name;
	buffer.access = BufferAccess::WriteOnly;
	buffer.zero_initialize = true;
	return buffer;
}

BufferBuilder Instance::makeReadWrite(const std::string& name)
{
	if (!pimpl)
		throw std::runtime_error("FlowVk: makeReadWrite on empty Instance");
	BufferBuilder buffer;
	buffer.owner = pimpl;
	buffer.name = name;
	buffer.access = BufferAccess::ReadWrite;
	buffer.zero_initialize = false;
	return buffer;
}

} // namespace Flow
