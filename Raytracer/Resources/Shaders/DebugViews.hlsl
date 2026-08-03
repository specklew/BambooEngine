#ifndef DEBUG_VIEWS_HLSL
#define DEBUG_VIEWS_HLSL

// Include after DebugData is declared. Only surface views live here — anything
// that reads a VXPG buffer is a graph node now (DebugViewPass, ADR 0017 phase
// 5b), so it works under every technique instead of rasterization alone.

float4 DebugView_Surface(int mode, DebugData d)
{
    if (mode == 1) return d.albedo;
    if (mode == 2) return float4(d.worldNormal * 0.5 + 0.5, 1);
    if (mode == 3) return float4(d.vertexNormal * 0.5 + 0.5, 1);
    if (mode == 4) return d.normalMap;
    if (mode == 5) return float4(d.tangent * 0.5 + 0.5, 1);
    if (mode == 6) return float4(d.uv, 0, 1);
    if (mode == 7) return float4(d.roughness, d.metallic, 0, 1);
    if (mode == 13)
    {
        // Surface NaN sources. Red = shading normal, green = tangent parallel to N, blue = vertex normal.
        float parallel = saturate((d.tangentDotN - 0.9) * 10.0);
        return float4(d.worldNormalNaN, parallel, d.normalNaN, 1);
    }
    return float4(-1, -1, -1, -1);
}

bool TryDebugView(int mode, DebugData d, out float4 color)
{
    color = float4(0, 0, 0, 1);

    float4 surface = DebugView_Surface(mode, d);
    if (surface.x >= 0)
    {
        color = surface;
        return true;
    }
    return false;
}

#endif
