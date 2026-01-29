#include "../include/flowVk/Buffer.hpp"
#include "../include/flowVk/Instance.hpp"
#include "internal/InstanceImpl.hpp"

#include <stdexcept>
#include <cstring>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace Flow {

static VkBufferUsageFlags ssbo_usage()
{
	return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
}

static InstanceImpl::BufferState& get_state(const Buffer& buffer)
{
	if (!buffer.owner)
		throw std::runtime_error("FlowVk: Buffer has no owner");
	auto it = buffer.owner->buffers.find(buffer.name);
	if (it == buffer.owner->buffers.end())
		throw std::runtime_error("FlowVk: Unknown buffer name: " + buffer.name);
	return it->second;
}
static const InstanceImpl::BufferState& get_state_const(const Buffer& buffer)
{
	return get_state(const_cast<Buffer&>(buffer));
}

std::size_t Buffer::sizeBytes() const
{
	return get_state_const(*this).sizeBytes;
}

BufferAccess Buffer::access() const
{
	return get_state_const(*this).access;
}

void Buffer::setBytes(const void* data, std::size_t bytes)
{
	auto& state = get_state(*this);
	if (!state.buffer)
		throw std::runtime_error("FlowVk: setBytes on unallocated buffer");
	if (bytes > state.sizeBytes)
		throw std::runtime_error("FlowVk: setBytes exceeds buffer size");

	void* mapped = nullptr;
	vmaMapMemory(owner->allocator, state.allocation, &mapped);
	std::memcpy(mapped, data, bytes);
	vmaUnmapMemory(owner->allocator, state.allocation);
}

void Buffer::getBytes(void* out, std::size_t bytes) const
{
	const auto& state = get_state_const(*this);
	if (!state.buffer)
		throw std::runtime_error("FlowVk: getBytes on unallocated buffer");
	if (bytes > state.sizeBytes)
		throw std::runtime_error("FlowVk: getBytes exceeds buffer size");

	void* mapped = nullptr;
	vmaMapMemory(owner->allocator, state.allocation, &mapped);
	std::memcpy(out, mapped, bytes);
	vmaUnmapMemory(owner->allocator, state.allocation);
}

static void alloc_or_resize(InstanceImpl* pimpl, InstanceImpl::BufferState& state, std::size_t bytes)
{
	if (bytes == 0)
		return;
	if (state.sizeBytes == bytes && state.buffer != VK_NULL_HANDLE)
		return;

	if (state.buffer)
	{
		vmaDestroyBuffer(pimpl->allocator, state.buffer, state.allocation);
		state.buffer = VK_NULL_HANDLE;
		state.allocation = VK_NULL_HANDLE;
	}

	VkBufferCreateInfo bufferCreateInfo{};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.size = bytes;
	bufferCreateInfo.usage = ssbo_usage();
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo allocationCreateInfo{};
	allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
	allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
					             VMA_ALLOCATION_CREATE_MAPPED_BIT;

	VkResult r = vmaCreateBuffer(pimpl->allocator, &bufferCreateInfo, &allocationCreateInfo, &state.buffer, &state.allocation, nullptr);
	if (r != VK_SUCCESS)
		throw std::runtime_error("FlowVk: vmaCreateBuffer failed");
	state.sizeBytes = bytes;
}

static void ensure_buffer_state(InstanceImpl* pimpl, const std::string& name, BufferAccess access)
{
	if (name.empty())
		throw std::runtime_error("FlowVk: buffer name must not be empty");

	auto it = pimpl->buffers.find(name);
	if (it == pimpl->buffers.end())
	{
		InstanceImpl::BufferState state{};
		state.name = name;
		state.access = access;
		pimpl->buffers.emplace(name, std::move(state));
		return;
	}

	if (it->second.access != access)
		throw std::runtime_error("FlowVk: buffer '" + name + "' already exists with different access");
}

Buffer BufferBuilder::allocateBytes(std::size_t bytes) const
{
	if (!owner)
		throw std::runtime_error("FlowVk: BufferBuilder has no owner");
	ensure_buffer_state(owner.get(), name, access);

	auto& state = owner->buffers.at(name);
	alloc_or_resize(owner.get(), state, bytes);

	Buffer buffer;
	buffer.owner = owner;
	buffer.name = name;
	return buffer;
}

void Buffer::zeroFill()
{
	if (!owner)
		throw std::runtime_error("FlowVk: zeroFill on empty Buffer");
	auto& state = get_state(*this);
	if (!state.buffer)
		throw std::runtime_error("FlowVk: zeroFill requires allocated buffer");

	owner->submit_one_time([&](VkCommandBuffer cmd) {
		vkCmdFillBuffer(cmd, state.buffer, 0, state.sizeBytes, 0);

		VkBufferMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer = state.buffer;
		barrier.offset = 0;
		barrier.size = state.sizeBytes;

		vkCmdPipelineBarrier(cmd,
		  VK_PIPELINE_STAGE_TRANSFER_BIT,
		  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		  0, 0, nullptr, 1, &barrier, 0, nullptr);
	});
}

void Buffer::resizeBytes(std::size_t newSizeBytes, bool zeroInit)
{
	if (!owner)
		throw std::runtime_error("FlowVk: resizeBytes on empty Buffer");
	auto& state = get_state(*this);
	alloc_or_resize(owner.get(), state, newSizeBytes);
	if (zeroInit)
		zeroFill();
}

} // namespace Flow
