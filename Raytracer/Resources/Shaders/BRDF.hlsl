#ifndef BRDF_HLSL
#define BRDF_HLSL

static const float PI = 3.14159265359;

// ---- Cook-Torrance BRDF components ----

float DistributionGGX(float NdotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(float NdotV, float NdotL, float roughness)
{
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

// ---- Tangent-space utilities ----

void BuildONB(float3 N, out float3 T, out float3 B)
{
    float3 up = abs(N.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

float3 TangentToWorld(float3 dir, float3 N, float3 T, float3 B)
{
    return dir.x * T + dir.y * B + dir.z * N;
}

// ---- Importance sampling ----

float3 ImportanceSampleGGX(float2 xi, float3 N, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;

    float phi = 2.0 * PI * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    float3 H_tangent = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    float3 T, B;
    BuildONB(N, T, B);
    return TangentToWorld(H_tangent, N, T, B);
}

float3 CosineSampleHemisphere(float2 xi, float3 N)
{
    float phi = 2.0 * PI * xi.x;
    float cosTheta = sqrt(1.0 - xi.y);
    float sinTheta = sqrt(xi.y);

    float3 dir = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    float3 T, B;
    BuildONB(N, T, B);
    return TangentToWorld(dir, N, T, B);
}

#endif // BRDF_HLSL