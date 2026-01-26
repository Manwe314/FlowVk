#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>

namespace Flow {

struct InstanceConfig {
	std::vector<const char*> instance_extensions{};
	std::vector<const char*> device_extensions{};

	std::string prefer_device_name_contains{};

	bool enable_validation = false;
};

struct Instance {
	struct Impl;
	std::shared_ptr<Impl> pimpl{};

	explicit operator bool() const noexcept { return static_cast<bool>(pimpl); }
	void addKernel(const std::string& kernelName, const std::filesystem::path& spvPath);
};

Instance makeInstance(const InstanceConfig& config = {});

} // namespace Flow
