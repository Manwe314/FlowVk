#pragma once
#include <cstdint>
#include <string_view>
#include <span>

namespace FlowVk::shader_meta {

enum class Access : uint8_t { ReadOnly, WriteOnly, ReadWrite };
enum class Layout : uint8_t { Std430, Std140, Scalar, Unknown };

struct BufferBinding {
	std::string_view name;
	std::string_view type_name;
	Access access;
	Layout layout;
	uint32_t set;
	uint32_t binding;
};

struct Module {
	std::string_view kernel_name;
	std::span<const BufferBinding> buffers;
};

} // namespace FlowVk::shader_meta
