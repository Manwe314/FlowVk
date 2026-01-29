# FlowVk

FlowVk is a minimal Vulkan compute wrapper focused on named storage buffers and simple kernel dispatch.
![logo](https://github.com/Manwe314/FlowVk/blob/main/images/FlowIcon.png "Flow logo")

## Table of content

- [Quick Start](#quick-start)
- [Dependencies and prerequisites](#dependencies-and-prerequisites)
- [About the library](#about-the-library)
	- [ABI and API compatibility](#abi-and-api-compatibility)
- [Public API](#public-api)
	- [struct Flow::InstanceConfig](#struct-flowinstanceconfig)
	- [struct Flow::Instance](#struct-flowinstance)
	- [struct Flow::BufferBuilder](#struct-flowbufferbuilder)
	- [Flow::Instance makeInstance](#flowinstance-makeinstanceconst-instanceconfig-config--)
- [Example Use](#example-use)


## Quick start

1) Add FlowVk to your project (CMake):

```cmake
add_subdirectory(path/to/FlowVk)
add_executable(app main.cpp)

# Builds shaders, generates metadata, and links FlowVk.
flowvk_add_kernels(TARGET app SHADERS shaders/multiply.comp)
```

`flowvk_add_kernels` is the main function to get Flow working in your project

2) Use the API:

```cpp
#include <FlowVk.hpp>
#include <filesystem>
#include <vector>

int main() {
  Flow::Instance instance = Flow::makeInstance();

  instance.addKernel("multiply", "build/shaders/multiply.spv");

  std::vector<float> a = {1, 2, 3, 4};
  std::vector<float> b = {5, 6, 7, 8};

  auto bufA = instance.makeReadOnly("a").fromVector(a);
  auto bufB = instance.makeReadOnly("b").fromVector(b);
  auto bufOut = instance.makeWriteOnly("out").withSizeBytes(a.size() * sizeof(float));

  instance.runSingleKernel("multiply", /*local x*/ 64, /* workgroups y*/ 1, /* dimension z */, 1);

  const std::vector<float> result = bufOut.getValues<float>();
}
```

3) Write a compute shader using FlowVk decorators:

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

Important Notes:
- `addKernel`'s first argument, its name, **MUST** be the stem of the spv file 
> For example if you name your comp shader compute.comp the kernel name **must** be compute.

- `make*X*Only` functions must have their names, I.e the argument passed to them, match the `name=` defined in the shader. Their access types must match as well.

- `flowvk_add_kernels` cmake function generates all the necacery files and compiles the shaders using `glslc` into spv's.

## Dependencies and prerequisites

- C++23 compiler
- CMake 3.21+
- Vulkan SDK. you can grab it [here](https://vulkan.lunarg.com/sdk/home)
- `glslc` from Vulkan SDK if you use the shader pipeline
- Vulkan-capable GPU with a compute queue

Optional helper: `vulkan_env.sh` can source a Vulkan SDK `setup-env.sh` on your machine.
simply run `source path/to/sctipt/vulkan_env.sh`

Flow Expects you to set up Vulkan on your machine.
Flow Will always be mainly made for linux but with minimal extra work you should be able to run it on windows or mac.

## About the library

FlowVk provides:
- A small C++ API for loading SPIR-V compute kernels and dispatching them.
- Named, host-visible storage buffers backed by Vulkan Memory Allocator (VMA).
- A shader preprocessor (`FlowVk_ShaderPP`) that turns `@buffer[...]` decorators into GLSL
  SSBO declarations and generates metadata (`KernelBuffers.hpp`) so buffers can be matched by name.

Buffer binding metadata is derived from the shader filename stem and the order of `@buffer`
declarations (set = 0, binding increments).

### ABI and API compatibility

- FlowVk exposes STL types (`std::string`, `std::vector`, `std::shared_ptr`) in its public API,
  so the ABI is tied to your compiler and standard library.
- There is no stable cross-compiler ABI guarantee; build FlowVk with the same toolchain as your app.
- The library is currently a static library target (`FlowVk::FlowVk`).
- The API surface is intentionally minimal in this v1.0.0 baseline and may evolve.

## [Public API](include/flowVk/Instance.hpp)

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
  - `kernelName` must match a module in the kernel registry meaning it must match the shader files name (the stem only).
  - Throws `std::runtime_error` if the instance is empty, the kernel already exists, the registry
    is missing, or the SPIR-V is invalid.

- `void runSingleKernel(const std::string& kernelName, uint32_t groupCountX = 1, uint32_t groupCountY = 1, uint32_t groupCountZ = 1)`
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

See [Buffer.hpp](include/flowVk/Buffer.hpp) for buffer read/write helpers (`setBytes`, `getBytes`,
`getValues`, `resizeBytes`, and `zeroFill`).

### `Flow::Instance makeInstance(const InstanceConfig& config = {})`
Creates and initializes a Vulkan instance/device/queue and VMA allocator.

- Uses Vulkan API version 1.3.
- Throws `std::runtime_error` on failure (no compute device, extension failure, etc.).

## Example Use

in this example lets implement linear regression using copmute Pipeline with Flow.

lets start with cpp side. lets write the function signature and basic variables:

```cpp
void gpuTrain(const std::vector<float> dataPointsX, const std::vector<float> dataPointsY)
{
	double coefficentM;
	double constantB;
	int size = std::min(dataPointsX.size(), dataPointsY.size());
	int workgroupCount = static_cast<int>(std::ceil(static_cast<float>(size) / 64.0f));
	if (workgroupCount < 1)
		workgroupCount = 1;
	if (workgroupCount > 64)
		workgroupCount = 64;
}
```

Almost all of linnear regression work will be done by the Shader. without going to deep into the implementation of linnear regression our mission is to have 4 sums.
- sum of all X
- sum of all Y
- sum of all X * X
- sum of all X * Y
Gpu will help us here by summing members of vector X and Y faster.
without going too much into detail on how this is done on the gpu and focusing on Flow what we need in the shader is:
- a buffer that has the floats in data point X
- a buffer that has the floats in data point Y
- The number of total datapoints
- out put buffer that will retrun to the CPU the partial Sums for the final pass.

to get these resources lets use Flows GLSL decorations:

```GLSL
#version 460
layout(local_size_x = 64) in;

@buffer[name="numX" access="read_only" type="float" layout="std430"]

@buffer[name="numY" access="read_only" type="float" layout="std430"]

@buffer[name="size" access="read_only" type="int" layout="std430"]

@buffer[name="partials" access="write_only" type="vec4" layout="std430"]

shared vec4 shSums[64];
```
> We have to remember to use the same "name"-s for the buffers created on the CPP side. 
> size is a single int but Flow can only make SSBO Buffers **for now** hence we need to make a buffer for it.
> notice we set appropriate access and types for each resource. For vec4 on the CPP side we can use `glm::vec4` or a custom Vec4 Struct.

now that we have these buffers ready on the gpu lets finish the rest of the implementation of the shader side by writing the `main` function:

```GLSL
void main() {
    uint localId = gl_LocalInvocationID.x;
    uint globalId = gl_GlobalInvocationID.x;

    uint threads = gl_WorkGroupSize.x;
    uint totalThreads = gl_NumWorkGroups.x * threads;

    // (sumX, sumY, sumXX, sumXY)
    vec4 local = vec4(0.0);

    for (uint i = globalId; i < size.data[0]; i += totalThreads)
	{
        float x = numX.data[i];
        float y = numY.data[i];
        local.x += x;
        local.y += y;
        local.z += x * x;
        local.w += x * y;
    }

    shSums[localId] = local;
    barrier();

    // Tree reduction in shared memory (64 -> 1)
    for (uint offset = threads >> 1; offset > 0u; offset >>= 1u)
	{
        if (localId < offset)
            shSums[localId] += shSums[localId + offset];
        barrier();
    }

    // One write per workgroup (no atomics)
    if (localId == 0u)
        partials.data[gl_WorkGroupID.x] = shSums[0];
}
```

> note that using te buffers we must access their valuse using `.data[]` otherwise rest of the shader is clasic GLSL.

Now that we have the Shader Ready and we already named all the resources we are going to need lets implement all the GPU compute pipline work that needs to be done on the CPP side using Flow:

```cpp
// we initializie a flow instance with defaults.
Flow::Instance flow = Flow::makeInstance();

// we make the resources with the same names as we wrote in GLSL decor.
// in the same line we upload the data to numbers X and Y with .fromVector function
// and we size the out put buffer to be able to store the outputs of the shader using a custom vec4 struct.
Flow::Buffer numbersX = flow.makeReadOnly("numX").fromVector(dataPointsX);
Flow::Buffer numbersY = flow.makeReadOnly("numY").fromVector(dataPointsY);
Flow::Buffer dataPointsSize = flow.makeReadOnly("size").fromVector(std::vector<int>{size});
Flow::Buffer outPut = flow.makeWriteOnly("partials").withSizeBytes(static_cast<size_t>(sizeof(vec4) * workgroupCount));

// now we add the kernel making sure that the name matches the stem of .spv file.
// note that we build the executable into the build folder so we point to where spv ends up.
flow.addKernel("closedForm", "build/shaders/closedForm.spv");

// now we run the kernel, note that no buffer binding is needed, flow will check that the resources declared in GLSL exist in flows own registry and will throw a runtime error if any aremissing / missdeclared.
flow.runSingleKernel("closedForm", 64, workgroupCount, 1);
```
All of our Flow work is done now to finish the linnear regression part of our program we take the results from running the closedForm.spv which is in the outPut buffer and do one final step on the CPU side to get the final 2 values we need.

```cpp
const std::vector<vec4> result = outPut.getValues<vec4>();

float sumX{};
float sumY{};
float sumXX{};
float sumXY{};

for (auto& entry : result)
{
	sumX += entry.x;
	sumY += entry.y;
	sumXX += entry.z;
	sumXY += entry.w;
}

float denom = size * sumXX - sumX * sumX;

if (std::abs(denom) > 1e-20f)
{
	coefficentM = (size * sumXY - sumX * sumY) / denom;
	constantB = (sumY - coefficentM * sumX) / size;
}
else
{
	coefficentM = 0.0;
	constantB = sumY / size;
}
```

And now we have the M and B we want for predicting trends!
