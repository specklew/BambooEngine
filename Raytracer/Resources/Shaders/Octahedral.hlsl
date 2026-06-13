#ifndef OCTAHEDRAL_HLSL
#define OCTAHEDRAL_HLSL

// Octahedral normal encoding (32-bit unorm, two UNORM16 lanes).
// Ported from SIByLEngine2023 include/common/octahedral.hlsli (UE5 / ShaderToy
// derivation). Used to pack a unit normal into a single float (bit-cast) so a
// world-position + normal fit in one RGBA32F texel (the ShadingPoints G-buffer).

float2 OctWrap(float2 v)
{
    float2 w = 1.0 - abs(float2(v.y, v.x));
    if (v.x < 0.0) w.x = -w.x;
    if (v.y < 0.0) w.y = -w.y;
    return w;
}

float2 UnitVectorToSignedOctahedron(float3 n)
{
    n.xy /= dot(float3(1, 1, 1), abs(n));
    return (n.z < 0.0) ? OctWrap(n.xy) : n.xy;
}

float3 SignedOctahedronToUnitVector(float2 oct)
{
    float3 n = float3(oct, 1.0 - dot(float2(1, 1), abs(oct)));
    float t = max(-n.z, 0.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

uint UnitVectorToUnorm32Octahedron(float3 normal)
{
    float2 p = UnitVectorToSignedOctahedron(normal);
    p = clamp(p * 0.5 + 0.5, 0.0, 1.0);
    return uint(p.x * 0xfffe) | (uint(p.y * 0xfffe) << 16);
}

float3 Unorm32OctahedronToUnitVector(uint pUnorm)
{
    float2 p;
    p.x = clamp(float(pUnorm & 0xffff) / float(0xfffe), 0.0, 1.0);
    p.y = clamp(float(pUnorm >> 16) / float(0xfffe), 0.0, 1.0);
    p = p * 2.0 - 1.0;
    return SignedOctahedronToUnitVector(p);
}

#endif // OCTAHEDRAL_HLSL
