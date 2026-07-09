#ifndef SPHERICAL_QUAD_HLSL
#define SPHERICAL_QUAD_HLSL

// Spherical-quad (spherical rectangle) solid-angle sampling, Urena et al.
// "An Area-Preserving Parametrization for Spherical Rectangles" (EGSR 2013).
// Ported from SIByL raytracer/primitives/quad.hlsli (SphQuad / SphQuadInit /
// SampleSphQuad) + vxguiding_interface.hlsli CreateSphQuad. Used by the VXPG
// integrator to sample a voxel AABB's visible faces exactly by solid angle.

// SIByL SphQuad.
struct SphericalQuad
{
    float3 o, x, y, z;     // local reference system 'R'
    float z0, z0sq;
    float x0, y0, y0sq;    // rectangle coords in 'R'
    float x1, y1, y1sq;
    float b0, b1, b0sq, k; // misc precomputed constants
    float S;               // solid angle of the quad
};

// SIByL SphQuadInit. s = quad corner, ex/ey = edge vectors, o = reference point.
void SphericalQuadInit(float3 s, float3 ex, float3 ey, float3 o, out SphericalQuad squad)
{
    squad.o = o;
    float exl = length(ex);
    float eyl = length(ey);
    // compute local reference system 'R'
    squad.x = ex / exl;
    squad.y = ey / eyl;
    squad.z = cross(squad.x, squad.y);
    // compute rectangle coords in local reference system
    float3 d = s - o;
    squad.z0 = dot(d, squad.z);
    // flip 'z' to make it point against the quad
    if (squad.z0 > 0.0)
    {
        squad.z *= -1.0;
        squad.z0 *= -1.0;
    }
    squad.z0sq = squad.z0 * squad.z0;
    squad.x0 = dot(d, squad.x);
    squad.y0 = dot(d, squad.y);
    squad.x1 = squad.x0 + exl;
    squad.y1 = squad.y0 + eyl;
    squad.y0sq = squad.y0 * squad.y0;
    squad.y1sq = squad.y1 * squad.y1;
    // create vectors to four vertices
    float3 v00 = float3(squad.x0, squad.y0, squad.z0);
    float3 v01 = float3(squad.x0, squad.y1, squad.z0);
    float3 v10 = float3(squad.x1, squad.y0, squad.z0);
    float3 v11 = float3(squad.x1, squad.y1, squad.z0);
    // compute normals to edges
    float3 n0 = normalize(cross(v00, v10));
    float3 n1 = normalize(cross(v10, v11));
    float3 n2 = normalize(cross(v11, v01));
    float3 n3 = normalize(cross(v01, v00));
    // compute internal angles (gamma_i)
    float g0 = acos(-dot(n0, n1));
    float g1 = acos(-dot(n1, n2));
    float g2 = acos(-dot(n2, n3));
    float g3 = acos(-dot(n3, n0));
    // compute predefined constants
    squad.b0 = n0.z;
    squad.b1 = n2.z;
    squad.b0sq = squad.b0 * squad.b0;
    squad.k = 2.0 * PI - g2 - g3;
    // compute solid angle from internal angles
    squad.S = g0 + g1 - squad.k;
}

// Spherical quad for an axis-aligned rectangle spanning [-extend, +extend] in
// the local xy plane at z = 0, seen from reference point `local`.
// SIByL CreateSphQuad (float2-extend overload).
SphericalQuad CreateSphericalQuad(float3 local, float2 extend)
{
    const float2 extend2 = extend + extend;
    SphericalQuad squad;
    SphericalQuadInit(float3(-extend, 0), float3(extend2.x, 0, 0), float3(0, extend2.y, 0), local, squad);
    return squad;
}

// Draws a direction uniformly distributed over the quad's solid angle.
// SIByL SampleSphQuad.
void SampleSphericalQuad(
    float3 pos, SphericalQuad squad, float2 uv,
    out float3 w, out float pdf)
{
    const float eps = 0.0001;
    // 1. compute 'cu'
    float au = uv.x * squad.S + squad.k;
    float fu = (cos(au) * squad.b0 - squad.b1) / sin(au);
    float cu = 1.0 / sqrt(fu * fu + squad.b0sq) * (fu > 0.0 ? +1.0 : -1.0);
    cu = clamp(cu, -1.0, 1.0); // avoid NaNs
    // 2. compute 'xu'
    float xu = -(cu * squad.z0) / sqrt(1.0 - cu * cu);
    xu = clamp(xu, squad.x0, squad.x1); // avoid Infs
    // 3. compute 'yv'
    float d = sqrt(xu * xu + squad.z0sq);
    float h0 = squad.y0 / sqrt(d * d + squad.y0sq);
    float h1 = squad.y1 / sqrt(d * d + squad.y1sq);
    float hv = h0 + uv.y * (h1 - h0);
    float hv2 = hv * hv;
    float yv = (hv2 < 1.0 - eps) ? (hv * d) / sqrt(1.0 - hv2) : squad.y1;
    // 4. transform (xu, yv, z0) to world coords
    float3 p = squad.o + xu * squad.x + yv * squad.y + squad.z0 * squad.z;
    w = normalize(p - pos);
    pdf = 1.0 / squad.S;
}

#endif // SPHERICAL_QUAD_HLSL
