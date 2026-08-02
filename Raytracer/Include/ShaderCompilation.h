#pragma once

struct ShaderMetadata;

// The returned pointer must be owned/freed by the caller. Returns nullptr if failed to compile.
// outReflection, when given, receives DXC's reflection blob for the same compile — the input
// to ShaderReflection, and the only way to learn which registers the shader really touches.
Microsoft::WRL::ComPtr<IDxcBlob> CompileShader(const ShaderMetadata&             desc,
                                               Microsoft::WRL::ComPtr<IDxcBlob>* outReflection = nullptr);
