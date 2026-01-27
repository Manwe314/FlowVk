#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>

#include "Buffer.hpp"

namespace Flow {

struct InstanceImpl;

struct InstanceConfig {
	std::vector<const char*> instance_extensions{};
	std::vector<const char*> device_extensions{};

	std::string prefer_device_name_contains{};

	bool enable_validation = false;
};

struct BufferBuilder;

struct Instance {
	struct Impl;
	std::shared_ptr<InstanceImpl> pimpl{};

	explicit operator bool() const noexcept { return static_cast<bool>(pimpl); }
	void addKernel(const std::string& kernelName, const std::filesystem::path& spvPath);
	void runSingleKernel(const std::string& kernelName, uint32_t groupCountX = 1, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);
	BufferBuilder makeReadOnly(const std::string& name);
	BufferBuilder makeWriteOnly(const std::string& name);
	BufferBuilder makeReadWrite(const std::string& name);
};

Instance makeInstance(const InstanceConfig& config = {});

struct BufferBuilder {
	std::shared_ptr<InstanceImpl> owner;
	std::string name;
	BufferAccess access = BufferAccess::ReadOnly;

	bool zero_initialize = false; // RO/RW default false
	bool allow_resize = true;

	Buffer allocateBytes(std::size_t bytes) const;

	template<class T>
	Buffer fromVector(const std::vector<T>& vector) const
	{
		Buffer buffer = allocateBytes(vector.size() * sizeof(T));
		buffer.setValues(vector);
		return buffer;
	}

	Buffer withSizeBytes(std::size_t bytes, bool zeroInit = true) const
	{
		Buffer buffer = allocateBytes(bytes);
		if (zeroInit)
			buffer.zeroFill();
		return buffer;
	}

  	operator Buffer() const { return allocateBytes(0); }
};

} // namespace Flow
