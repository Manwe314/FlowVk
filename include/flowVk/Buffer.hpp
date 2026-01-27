#pragma once
#include <memory>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace Flow {

enum struct BufferAccess : uint8_t { ReadOnly, WriteOnly, ReadWrite };

struct BufferCreateInfo {
	std::size_t size_bytes = 0;
	bool zero_initialize = true;
};

struct InstanceImpl;

struct Buffer {
	std::shared_ptr<InstanceImpl> owner;
	std::string name;

	explicit operator bool() const noexcept { return owner && !name.empty(); }

	std::size_t sizeBytes() const;
	BufferAccess access() const;

	void resizeBytes(std::size_t newSizeBytes, bool zeroInit = false);

	template<class T>
	void setValues(const std::vector<T>& v)
	{
		setBytes(v.data(), v.size() * sizeof(T));
	}

	template<class T>
	std::vector<T> getValues() const
	{
		auto bytes = sizeBytes();
		if (bytes % sizeof(T) != 0)
			throw std::runtime_error("FlowVk: getValues<T> size mismatch");
		std::vector<T> out(bytes / sizeof(T));
		getBytes(out.data(), bytes);
		return out;
	}

	void setBytes(const void* data, std::size_t bytes);
	void getBytes(void* out, std::size_t bytes) const;
	void zeroFill();
};

} // namespace Flow
