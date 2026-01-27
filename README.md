# FlowVk

FlowVk is a minimal Vulkan compute wrapper focused on named storage buffers and simple kernel dispatch.

## Quick start

1) Add FlowVk to your project (CMake):

```cmake
add_subdirectory(path/to/FlowVk)
add_executable(app main.cpp)

# Builds shaders, generates metadata, and links FlowVk.
flowvk_add_kernels(TARGET app SHADERS shaders/multiply.comp)
```

`flowvk_add_kernels` will build the `FlowVk_ShaderPP` tool, generate `KernelBuffers.hpp`,
compile SPIR-V via `glslc`, and link `FlowVk::FlowVk`.

2) Write a compute shader using FlowVk decorators:

```glsl
// shaders/multiply.comp
#version 460
@buffer[name="a",   access=read_only,  type=float, layout=std430]
@buffer[name="b",   access=read_only,  type=float, layout=std430]
@buffer[name="out", access=write_only, type=float, layout=std430]

layout(local_size_x = 64) in;
void main() {
  uint i = gl_GlobalInvocationID.x;
  out.data[i] = a.data[i] * b.data[i];
}
```

3) Use the API:

```cpp
#include <FlowVk.hpp>
#include <filesystem>
#include <vector>

int main() {
  Flow::Instance instance = Flow::makeInstance({
    .enable_validation = false
  });

  // Kernel name is the shader filename stem: "multiply"
  instance.addKernel("multiply", std::filesystem::path("shaders/multiply.spv"));

  std::vector<float> a = {1, 2, 3, 4};
  std::vector<float> b = {5, 6, 7, 8};

  auto bufA = instance.makeReadOnly("a").fromVector(a);
  auto bufB = instance.makeReadOnly("b").fromVector(b);
  auto bufOut = instance.makeWriteOnly("out").withSizeBytes(a.size() * sizeof(float));

  instance.runSingleKernel("multiply", static_cast<uint32_t>((a.size() + 63) / 64));

  auto result = bufOut.getValues<float>();
  (void)result;
}
```

Notes:
- `addKernel` and `runSingleKernel` require the kernel registry (generated `KernelBuffers.hpp`)
  and the `FLOWVK_WITH_KERNEL_REGISTRY` compile definition. `flowvk_add_kernels` sets this up.
- The SPIR-V file must be available at runtime (copy it or install it next to the executable).

## Dependencies and prerequisites

- C++23 compiler
- CMake 3.21+
- Vulkan SDK (headers + loader)
- `glslc` from Vulkan SDK if you use the shader pipeline
- Vulkan-capable GPU with a compute queue
- Vulkan Memory Allocator (VMA) header is vendored at `include/external/vk_mem_alloc.h`

Optional helper: `vulkan_env.sh` can source a Vulkan SDK `setup-env.sh` on your machine.

## About the library

FlowVk provides:
- A small C++ API for loading SPIR-V compute kernels and dispatching them.
- Named, host-visible storage buffers backed by Vulkan Memory Allocator (VMA).
- A shader preprocessor (`FlowVk_ShaderPP`) that turns `@buffer[...]` decorators into GLSL
  SSBO declarations and generates metadata (`KernelBuffers.hpp`) so buffers can be matched by name.

Buffer binding metadata is derived from the shader filename stem and the order of `@buffer`
declarations (set = 0, binding increments).

## ABI and API compatibility

- FlowVk exposes STL types (`std::string`, `std::vector`, `std::shared_ptr`) in its public API,
  so the ABI is tied to your compiler and standard library.
- There is no stable cross-compiler ABI guarantee; build FlowVk with the same toolchain as your app.
- The library is currently a static library target (`FlowVk::FlowVk`).
- The API surface is intentionally minimal in this v1.0.0 baseline and may evolve.

## Public API (include/flowVk/Instance.hpp)

### `struct Flow::InstanceConfig`
Configuration for `makeInstance`.

- `std::vector<const char*> instance_extensions`
  - Vulkan instance extensions to enable. If empty, FlowVk uses its defaults (currently none).
- `std::vector<const char*> device_extensions`
  - Vulkan device extensions to enable. If empty, FlowVk uses its defaults (currently none).
- `std::string prefer_device_name_contains`
  - If non-empty, FlowVk prefers a physical device whose name contains this substring.
- `bool enable_validation`
  - Reserved for validation support. Currently not wired to any layers.

### `struct Flow::Instance`
Main entry point for runtime usage. Holds a shared internal implementation (`pimpl`).

- `explicit operator bool() const noexcept`
  - Returns true if the instance is valid.

- `void addKernel(const std::string& kernelName, const std::filesystem::path& spvPath)`
  - Loads a SPIR-V compute shader and creates a compute pipeline.
  - `kernelName` must match a module in the kernel registry (generated from shaders).
  - Throws `std::runtime_error` if the instance is empty, the kernel already exists, the registry
    is missing, or the SPIR-V is invalid.

- `void runSingleKernel(const std::string& kernelName,
                        uint32_t groupCountX = 1,
                        uint32_t groupCountY = 1,
                        uint32_t groupCountZ = 1)`
  - Dispatches a single compute kernel with the given workgroup counts.
  - Requires all buffers declared by the shader metadata to exist and be allocated.
  - Execution is synchronous; it waits for completion before returning.
  - Throws `std::runtime_error` on invalid instance, unknown kernel, missing buffers,
    or missing registry.

- `BufferBuilder makeReadOnly(const std::string& name)`
- `BufferBuilder makeWriteOnly(const std::string& name)`
- `BufferBuilder makeReadWrite(const std::string& name)`
  - Creates a `BufferBuilder` preconfigured with the requested access.
  - `name` must match the buffer name used in shader metadata.
  - Throws `std::runtime_error` if the instance is empty.

### `Flow::Instance makeInstance(const InstanceConfig& config = {})`
Creates and initializes a Vulkan instance/device/queue and VMA allocator.

- Uses Vulkan API version 1.3.
- Throws `std::runtime_error` on failure (no compute device, extension failure, etc.).

### `struct Flow::BufferBuilder`
Fluent helper for allocating buffers owned by an instance.

Public fields:
- `std::shared_ptr<InstanceImpl> owner`
- `std::string name`
- `BufferAccess access`
- `bool zero_initialize`
- `bool allow_resize`

Notes: `zero_initialize` and `allow_resize` are currently informational; allocation behavior
is controlled by the methods below.

Methods:
- `Buffer allocateBytes(std::size_t bytes) const`
  - Allocates (or resizes) the named buffer to `bytes` and returns a `Buffer` handle.
- `template<class T> Buffer fromVector(const std::vector<T>& vector) const`
  - Allocates enough space for `vector`, writes its contents, returns the `Buffer`.
- `Buffer withSizeBytes(std::size_t bytes, bool zeroInit = true) const`
  - Allocates to `bytes`, optionally zero-fills the buffer.
- `operator Buffer() const`
  - Shorthand for `allocateBytes(0)` (creates a handle without allocating).

See `include/flowVk/Buffer.hpp` for buffer read/write helpers (`setBytes`, `getBytes`,
`getValues`, `resizeBytes`, and `zeroFill`).
