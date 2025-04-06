#pragma once

struct ShaderMetadata;

// The returned pointer must be owned/freed by the caller. Returns nullptr if failed to compile.
Microsoft::WRL::ComPtr<IDxcBlob> CompileShader(const ShaderMetadata& desc);
