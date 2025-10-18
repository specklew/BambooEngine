#include "Common.hlsl"

struct STriVertex
{
    float3 vertex;
    //float4 color;
};

//StructuredBuffer<STriVertex> BTriVertex : register(t0);
//StructuredBuffer<int> indices : register(t1);

[shader("closesthit")] 
void Hit(inout HitInfo payload : SV_RayPayload, Attributes attrib) 
{
    float3 barycentrics = 
    float3(1.f - attrib.bary.x - attrib.bary.y, attrib.bary.x, attrib.bary.y);

    //uint vertId = 3 * PrimitiveIndex();
    //float3 hitColor = BTriVertex[indices[vertId + 0]].color * barycentrics.x +
               //BTriVertex[indices[vertId + 1]].color * barycentrics.y +
               //BTriVertex[indices[vertId + 2]].color * barycentrics.z;
  
    payload.colorAndDistance = float4(barycentrics.x,barycentrics.y,barycentrics.z,1); 
}
