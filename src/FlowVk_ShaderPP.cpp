#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <stdexcept>
#include <cstddef>
#include <algorithm>
#include <unordered_map>
#include <array>
#include <span>
#include <cctype>

#include "../include/flowVk/ShaderMeta.hpp"

enum class DecorKind { Buffer, PushConstant };

struct FoundDecor {
  DecorKind kind{};
  std::size_t position{};
  std::size_t token_length{};
};

static constexpr std::string_view bufferToken = "@buffer[";
static constexpr std::string_view pushToken   = "@push_constant[";

struct Args {
  std::filesystem::path in_file;
  std::filesystem::path out_glsl;
  std::filesystem::path out_hpp;
};

static void print_usage() {
  std::cout
    << "FlowVk_ShaderPP\n"
    << "Usage:\n"
    << "  FlowVk_ShaderPP --in <input.comp> --out-glsl <output.glsl> --out-hpp <output.hpp>\n";
}

static Args parse_args(int argc, char* argv[])
{
	if (argc < 7)
	{
		print_usage();
		throw std::runtime_error("FlowVk_ShaderPP: Incorrect arguments");
	}

	Args arguments{};
	int i = 1;
	while (i < argc)
	{
		const std::string_view arg = argv[i];

		if (arg == "--in")
		{
			if (i + 1 >= argc)
				throw std::runtime_error("FlowVk_ShaderPP: --in missing a value");
			arguments.in_file = std::filesystem::path(std::string_view{argv[i + 1]});
			i += 2;
			continue;
		}
		else if (arg == "--out-glsl")
		{
			if (i + 1 >= argc)
				throw std::runtime_error("FlowVk_ShaderPP: --out-glsl missing a value");
			arguments.out_glsl = std::filesystem::path(std::string_view{argv[i + 1]});
			i += 2;
			continue;
		}
		else if (arg == "--out-hpp")
		{
			if (i + 1 >= argc)
				throw std::runtime_error("FlowVk_ShaderPP: --out-hpp missing a value");
			arguments.out_hpp = std::filesystem::path(std::string_view{argv[i + 1]});
			i += 2;
			continue;
		}
		else
		  throw std::runtime_error(std::string("FlowVk_ShaderPP: Unknown argument: ") + std::string(arg));
	}

	if (arguments.in_file.empty() || arguments.out_glsl.empty() || arguments.out_hpp.empty())
	{
		print_usage();
		throw std::runtime_error("FlowVk_ShaderPP: missing required arguments");
	}
	return arguments;
}

static bool read_file_to_string(const std::filesystem::path& path, std::string& out)
{
	std::ifstream file(path, std::ios::binary);
	if (!file) return false;

	file.seekg(0, std::ios::end);
	const auto size = file.tellg();
	if (size < 0) return false;

	out.resize(static_cast<size_t>(size));
	file.seekg(0, std::ios::beg);
	file.read(out.data(), static_cast<std::streamsize>(out.size()));
	return static_cast<bool>(file);
}

static bool ensure_parent_dir(const std::filesystem::path& path)
{
	const auto parent = path.parent_path();
	if (parent.empty())
		return true;

	std::error_code ec;
	std::filesystem::create_directories(parent, ec);
	return !ec;
}

static bool write_string_to_file(const std::filesystem::path& path, const std::string& string)
{
	if (!ensure_parent_dir(path))
		return false;
	
	std::ofstream file(path, std::ios::binary);
	
	if (!file)
		return false;
	
	file.write(string.data(), static_cast<std::streamsize>(string.size()));
	return static_cast<bool>(file);
}

static bool find_next_decor(const std::string& string, std::size_t from, FoundDecor& out)
{
	const std::size_t positionBuffer = string.find(bufferToken, from);
	const std::size_t positionPush   = string.find(pushToken, from);

	if (positionBuffer == std::string::npos && positionPush == std::string::npos)
    	return false;

	const bool bufIsEarlier = (positionBuffer != std::string::npos) && (positionPush == std::string::npos || positionBuffer < positionPush);

	if (bufIsEarlier)
		out = {DecorKind::Buffer, positionBuffer, bufferToken.size()};
	else
		out = {DecorKind::PushConstant, positionPush, pushToken.size()};

	return true;
}

static std::optional<std::size_t> find_matching_bracket(const std::string& string, std::size_t open_pos)
{
	bool in_string = false;
	bool escaped = false;

	for (std::size_t i = open_pos; i < string.size(); ++i)
	{
		char c = string[i];

    	if (escaped)
		{
			escaped = false;
			continue;
		}
    	if (c == '\\')
		{
			escaped = true;
			continue;
		}

    	if (c == '"')
		{
			in_string = !in_string;
			continue;
		}
    	if (!in_string && c == ']')
			return i;
  	}
	return std::nullopt;
}

static void skip_whiteSpace(std::string_view stringView, std::size_t& i)
{
	while (i < stringView.size() && std::isspace(static_cast<unsigned char>(stringView[i])))
		++i;
}

static bool is_ident_char(char c)
{
	return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-';
}

static std::optional<std::string_view> parse_key(std::string_view stringView, std::size_t& i) {
	skip_whiteSpace(stringView, i);
	const std::size_t start = i;
	while (i < stringView.size() && is_ident_char(stringView[i]))
		++i;
	if (start == i)
		return std::nullopt;
	return stringView.substr(start, i - start);
}

static bool consume_char(std::string_view stringView, std::size_t& i, char expected)
{
	skip_whiteSpace(stringView, i);
  	if (i < stringView.size() && stringView[i] == expected)
	{
		++i;
		return true;
	}
	return false;
}

static std::optional<std::string> parse_value(std::string_view stringView, std::size_t& i) {
	skip_whiteSpace(stringView, i);
	if (i >= stringView.size())
		return std::nullopt;

	if (stringView[i] == '"')
	{
		++i;
		std::string out;
		bool escaped = false;
		while (i < stringView.size())
		{
	  		char c = stringView[i++];
	  		if (escaped)
			{
				out.push_back(c);
				escaped = false;
				continue;
			}
	  		if (c == '\\')
			{
				escaped = true;
				continue;
			}
	  		if (c == '"')
				return out;
	  		out.push_back(c);
		}
    	return std::nullopt;
	}

	const std::size_t start = i;
	while (i < stringView.size() && !std::isspace(static_cast<unsigned char>(stringView[i])))
		++i;
	if (start == i)
		return std::nullopt;
	return std::string(stringView.substr(start, i - start));
}

static std::optional<std::unordered_map<std::string, std::string>> parse_kv_pairs(std::string_view inner)
{
	std::unordered_map<std::string, std::string> kv;
	std::size_t i = 0;

	while (true)
	{
		skip_whiteSpace(inner, i);
		if (i >= inner.size())
			break;

		auto k = parse_key(inner, i);
		if (!k)
			return std::nullopt;

		if (!consume_char(inner, i, '='))
			return std::nullopt;

		auto v = parse_value(inner, i);
		if (!v)
			return std::nullopt;
		kv.emplace(std::string(*k), *v);
	}

  	return kv;
}

static std::string sanitize_cpp_ident(std::string_view stringView)
{
	std::string out;
	out.reserve(stringView.size());
	for (char c : stringView)
	{
		if (std::isalnum(static_cast<unsigned char>(c)) || c == '_')
			out.push_back(c);
		else
			out.push_back('_');
	}
	if (out.empty() || std::isdigit(static_cast<unsigned char>(out[0])))
		out.insert(out.begin(), '_');
	return out;
}

static std::string escape_cpp_string(std::string_view stringView)
{
	std::string out;
	out.reserve(stringView.size() + 8);
	for (char c : stringView)
	{
    	if (c == '\\')
			out += "\\\\";
    	else if (c == '"')
			out += "\\\"";
    	else
			out.push_back(c);
	}
	return out;
}

static std::string pascal_case(std::string_view s)
{
	std::string out;
	bool cap = true;
	for (char c : s)
	{
    	if (std::isalnum(static_cast<unsigned char>(c)))
		{
    	  out.push_back(cap ? static_cast<char>(std::toupper(static_cast<unsigned char>(c))) : c);
    	  cap = false;
		}
   		else
   		  cap = true;
  	}
  	if (out.empty())
		out = "Buffer";
  	if (std::isdigit(static_cast<unsigned char>(out[0])))
		out.insert(out.begin(), 'B');
  	return out;
}

struct BufferInfo {
	std::string name;
	std::string access;
	std::string type;
	std::string layout;
	uint32_t set = 0;
	uint32_t binding = 0;
};

static std::optional<std::string> access_to_glsl_qual(const std::string& s)
{
	if (s == "read_only" || s == "readonly" || s == "read-only") return "readonly ";
	if (s == "write_only" || s == "writeonly" || s == "write-only") return "writeonly ";
	if (s == "read_write" || s == "readwrite" || s == "read-write") return ""; // no qualifier
	return std::nullopt;
}

static std::string access_to_cpp_enum(const std::string& s)
{
	if (s == "read_only" || s == "readonly" || s == "read-only") return "Flow::shader_meta::Access::ReadOnly";
	if (s == "write_only" || s == "writeonly" || s == "write-only") return "Flow::shader_meta::Access::WriteOnly";
	return "Flow::shader_meta::Access::ReadWrite";
}

static std::string layout_to_cpp_enum(const std::string& s)
{
	if (s == "std430") return "Flow::shader_meta::Layout::Std430";
	if (s == "std140") return "Flow::shader_meta::Layout::Std140";
	if (s == "scalar") return "Flow::shader_meta::Layout::Scalar";
	return "Flow::shader_meta::Layout::Unknown";
}

static bool is_supported_layout(const std::string& s)
{
	return s == "std430" || s == "std140" || s == "scalar";
}

static std::string make_glsl_ssbo_decl(const BufferInfo& b)
{
	const std::string accessQual = access_to_glsl_qual(b.access).value_or("");
	const std::string blockName = pascal_case(b.name) + "Buffer";

	std::string out;
	out += "layout(set = " + std::to_string(b.set)
    	  + ", binding = " + std::to_string(b.binding)
    	  + ", " + b.layout + ") ";
	out += accessQual;
	out += "buffer " + blockName + " {\n";
	out += "  " + b.type + " data[];\n";
	out += "} " + b.name + ";\n";
	return out;
}

struct TransformResult {
	std::string out_glsl;
	std::vector<BufferInfo> buffers;
};

static TransformResult transform_shader(const std::string& text)
{
	std::unordered_map<std::string, std::size_t> name_to_index;
	std::vector<BufferInfo> buffers;
	uint32_t next_binding = 0;

	std::string out;
	out.reserve(text.size());

	std::size_t cursor = 0;
	std::size_t searchPos = 0;
	FoundDecor decor{};

	while (find_next_decor(text, searchPos, decor))
	{
    	out.append(text.substr(cursor, decor.position - cursor));

    	const std::size_t open_bracket_pos = decor.position + decor.token_length - 1;

		auto close_bracket = find_matching_bracket(text, open_bracket_pos);
		if (!close_bracket)
		{
			out += "/* FlowVk_ShaderPP ERROR: unterminated decoration */\n";
			cursor = decor.position + decor.token_length;
			searchPos = cursor;
			continue;
		}

		const std::size_t inner_start = open_bracket_pos + 1;
		const std::size_t inner_len = (*close_bracket) - inner_start;
		const std::string_view inner(text.data() + inner_start, inner_len);

		if (decor.kind == DecorKind::Buffer)
		{
    		auto kvOpt = parse_kv_pairs(inner);
    		if (!kvOpt)
    		  out += "/* FlowVk_ShaderPP ERROR: failed to parse @buffer[...] */\n";
    		else {
    			auto& kv = *kvOpt;

        		auto itName   = kv.find("name");
        		auto itAccess = kv.find("access");
        		auto itType   = kv.find("type");
        		auto itLayout = kv.find("layout");

        		if (itName == kv.end() || itAccess == kv.end() || itType == kv.end() || itLayout == kv.end())
          			out += "/* FlowVk_ShaderPP ERROR: @buffer requires name, access, type, layout */\n";
        		else {
          			const std::string& name   = itName->second;
          			const std::string& access = itAccess->second;
          			const std::string& type   = itType->second;
          			const std::string& layout = itLayout->second;

          			if (!access_to_glsl_qual(access).has_value()) {
          				out += "/* FlowVk_ShaderPP ERROR: access must be read_only/write_only/read_write */\n";
          			} else if (!is_supported_layout(layout)) {
          				out += "/* FlowVk_ShaderPP ERROR: layout must be std430/std140/scalar */\n";
          			} else {
            			auto it = name_to_index.find(name);
            			if (it == name_to_index.end()) {
            				BufferInfo bi;
            				bi.name = name;
            				bi.access = access;
            				bi.type = type;
            				bi.layout = layout;
            				bi.set = 0;
            				bi.binding = next_binding++;

            				name_to_index.emplace(name, buffers.size());
            				buffers.push_back(bi);

            				out += make_glsl_ssbo_decl(bi);
            			} else {
              				BufferInfo& existing = buffers[it->second];
              				const bool same = (existing.access == access && existing.type == type && existing.layout == layout);
              				if (!same) {
                				out += "/* FlowVk_ShaderPP ERROR: duplicate @buffer name with mismatched properties */\n";
              				} else {
                				// same buffer referenced again: emit nothing
              				}
            			}
          			}
				}
			}
		} else {
      		out += "/* FlowVk_ShaderPP: @push_constant not implemented yet */\n";
    	}

		// move past entire decoration
		cursor = (*close_bracket) + 1;
		searchPos = cursor;
	}

	out.append(text.substr(cursor));

	return TransformResult{std::move(out), std::move(buffers)};
}

static std::string emit_hpp(const std::filesystem::path& in_file, const std::vector<BufferInfo>& buffers)
{
	const std::string stem = sanitize_cpp_ident(in_file.stem().string());
	const std::string kernel_name = in_file.stem().string();

	std::string header;
	header += "#pragma once\n";
	header += "// Auto-generated by FlowVk_ShaderPP\n\n";
	header += "#include <array>\n";
	header += "#include <span>\n";
	header += "#include <string_view>\n";
	header += "#include <flowVk/ShaderMeta.hpp>\n\n";

	header += "namespace Flow::shader_meta::" + stem + " {\n\n";

	header += "inline constexpr std::array<Flow::shader_meta::BufferBinding, " + std::to_string(buffers.size()) + "> kBufferArray = {{\n";
	for (const auto& b : buffers)
	{
		header += "  Flow::shader_meta::BufferBinding{";
		header += "\"" + escape_cpp_string(b.name) + "\", ";
		header += "\"" + escape_cpp_string(b.type) + "\", ";
		header += access_to_cpp_enum(b.access) + ", ";
		header += layout_to_cpp_enum(b.layout) + ", ";
		header += std::to_string(b.set) + "u, ";
		header += std::to_string(b.binding) + "u";
		header += "},\n";
	}
	header += "}};\n\n";

	header += "inline constexpr Flow::shader_meta::Module module = {\n";
	header += "  .kernel_name = \"" + escape_cpp_string(kernel_name) + "\",\n";
	header += "  .buffers = std::span<const Flow::shader_meta::BufferBinding>(kBufferArray),\n";
	header += "};\n\n";

	header += "} // namespace Flow::shader_meta::" + stem + "\n";
	return header;
}

// ------------------------------------------------------------------

int main(int argc, char* argv[])
{
	Args args;
	try
	{
	args = parse_args(argc, argv);
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}

	std::string input;
	if (!read_file_to_string(args.in_file, input))
	{
		std::cerr << "Failed to read input file: " << args.in_file << "\n";
		return 2;
	}

	const TransformResult transformResult = transform_shader(input);

	if (!write_string_to_file(args.out_glsl, transformResult.out_glsl)) {
		std::cerr << "Failed to write GLSL output: " << args.out_glsl << "\n";
		return 3;
	}

	const std::string out_hpp = emit_hpp(args.in_file, transformResult.buffers);
	if (!write_string_to_file(args.out_hpp, out_hpp))
	{
		std::cerr << "Failed to write HPP output: " << args.out_hpp << "\n";
		return 4;
	}

	return 0;
}
