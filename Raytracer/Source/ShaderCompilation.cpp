#include "pch.h"
#include "ShaderCompilation.h"

#include <filesystem>
#include <sstream>

#include "Constants.h"
#include "Utils/Utils.h"
#include "Shader.h"


using Microsoft::WRL::ComPtr;

// Allocates memory
static wchar_t* ConvertToWideString(const char* narrowString)
{
	// TODO : what if invalid code page?
	const int size_needed = MultiByteToWideChar(CP_UTF8, 0, narrowString, -1, NULL, 0);
	wchar_t* wideString = new wchar_t[size_needed];
	MultiByteToWideChar(CP_UTF8, 0, narrowString, -1, wideString, size_needed);
	return wideString;
}

ComPtr<IDxcBlob> CompileShader(const ShaderMetadata& meta)
{
	// adapted from: https://github.com/microsoft/DirectXShaderCompiler/wiki/Using-dxc.exe-and-dxcompiler.dll

	static_assert(std::is_same_v<std::filesystem::path::value_type, wchar_t>);
	static_assert(std::is_same_v<LPCWSTR, const wchar_t*>);

	ComPtr<IDxcUtils> pUtils;
	ComPtr<IDxcCompiler3> pCompiler;
	ThrowIfFailed(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&pUtils)));
	ThrowIfFailed(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&pCompiler)));

	ComPtr<IDxcIncludeHandler> pIncludeHandler;
	ThrowIfFailed(pUtils->CreateDefaultIncludeHandler(&pIncludeHandler));

	const char* fullPath = FormatTempString("resources/%s", meta.szPathWithinResources);
	const std::unique_ptr<wchar_t> path = std::unique_ptr<wchar_t>(ConvertToWideString(fullPath));
	const std::unique_ptr<wchar_t> entrypoint = std::unique_ptr<wchar_t>(ConvertToWideString(meta.szEntrypoint));
	const std::unique_ptr<wchar_t> target = std::unique_ptr<wchar_t>(ConvertToWideString(meta.szTarget));
	// szDefines holds space-separated DXC tokens ("-D MyDefine=1 -D another=2").
	// Each must reach Compile() as its own argument — a single string with
	// internal spaces is one malformed option DXC silently mis-parses. Tokens
	// are collected fully before args takes c_str() pointers (vector growth
	// moves the strings).
	std::vector<std::wstring> userDefineTokens;
	if (meta.szDefines[0] != '\0')
	{
		const std::unique_ptr<wchar_t> defines(ConvertToWideString(meta.szDefines));
		std::wistringstream stream(defines.get());
		std::wstring token;
		while (stream >> token)
			userDefineTokens.push_back(token);
	}
	wchar_t standardDefines[64]{};
	const int charsWritten = swprintf(standardDefines, _countof(standardDefines), L"-DNUM_TEXTURES=%d", Constants::Graphics::MAX_TEXTURES);
	assert(charsWritten > 0);

	std::vector<LPCWSTR> args = {
		path.get(),
		L"-E", entrypoint.get(),
		L"-T", target.get(),
		L"-Zi", // enable debugging
		L"-Qembed_debug", // embed pdb into shader
		L"-I", L"Include",
		standardDefines,
	};

	// Native uint16_t (VXPG light-tree nodes, ADR 0003) needs -enable-16bit-types,
	// which DXC only accepts for shader model >= 6.2. Gate on the target model so
	// the SM 5.x raster shaders (colorShader etc.) still compile.
	{
		int major = 0, minor = 0;
		const char* underscore = meta.szTarget ? strchr(meta.szTarget, '_') : nullptr;
		if (underscore && sscanf_s(underscore + 1, "%d_%d", &major, &minor) == 2 &&
			(major > 6 || (major == 6 && minor >= 2)))
		{
			args.push_back(L"-enable-16bit-types");
		}
	}

	for (const std::wstring& token : userDefineTokens)
	{
		args.push_back(token.c_str());
	}

	// Open source file.
	ComPtr<IDxcBlobEncoding> pSource = nullptr;
	ThrowIfFailed(pUtils->LoadFile(path.get(), nullptr, &pSource));
	assert(pSource != nullptr && "Loading file failed. Make sure the file can be found.");
	DxcBuffer sourceCode{};
	sourceCode.Ptr = pSource->GetBufferPointer();
	sourceCode.Size = pSource->GetBufferSize();
	sourceCode.Encoding = DXC_CP_ACP; // Assume BOM says UTF8 or UTF16 or this is ANSI text.

	// Compile it with specified arguments
	ComPtr<IDxcResult> pResults;
	assert(args.size() < std::numeric_limits<uint32_t>::max());
	ThrowIfFailed(pCompiler->Compile(&sourceCode, args.data(), (UINT32)args.size(), pIncludeHandler.Get(), IID_PPV_ARGS(&pResults)));

	// Print errors if present
	ComPtr<IDxcBlobUtf8> pErrors = nullptr;
	ThrowIfFailed(pResults->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&pErrors), nullptr));
	if (pErrors != nullptr && pErrors->GetStringLength() != 0)
	{
		SPDLOG_WARN("Shader compilation ({}) -- warnings and errors: \n{}", fullPath, (const char*)pErrors->GetStringPointer());
	}

	// Quit if the compilation failed
	HRESULT hrStatus;
	ThrowIfFailed(pResults->GetStatus(&hrStatus));
	if (FAILED(hrStatus))
	{
		SPDLOG_ERROR("Compilation Failed ({}), ({})", fullPath, meta.szEntrypoint);
		return nullptr;
	}

	// Get shader bytecode
	ComPtr<IDxcBlob> pShader = nullptr;
	ThrowIfFailed(pResults->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&pShader), nullptr));
	assert(pShader != nullptr);

	return pShader;
}
