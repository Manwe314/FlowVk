#include "../include/flowVk/Instance.hpp"

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

struct Instance::Impl {
	VkInstance 		 instance = VK_NULL_HANDLE;
	VkPhysicalDevice physical = VK_NULL_HANDLE;
	VkDevice 		 device   = VK_NULL_HANDLE;

	uint32_t computeQueueFamily = UINT32_MAX;
	VkQueue  computeQueue       = VK_NULL_HANDLE;

	VmaAllocator allocator = VK_NULL_HANDLE;
	
	struct KernelState {
		VkShaderModule shaderModule = VK_NULL_HANDLE;
		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
		VkPipeline pipeline = VK_NULL_HANDLE;
		std::vector<VkDescriptorSetLayout> setLayouts;
	};

	std::unordered_map<std::string, KernelState> kernels;

	~Impl()
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
		
		if (allocator)	vmaDestroyAllocator(allocator);
		if (device)		vkDestroyDevice(device, nullptr);
		if (instance)	vkDestroyInstance(instance, nullptr);
	}
};

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

Instance makeInstance(const InstanceConfig& config)
{
	auto pimpl = std::make_shared<Instance::Impl>();

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

	const auto& mod = FlowVk::shader_meta::registry::get_module(kernelName);

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

	Instance::Impl::KernelState kernel{};

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

} // namespace Flow
