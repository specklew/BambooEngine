// Reduces the per-pixel Welford accumulator to two scalars: the mean variance of
// the accumulated ESTIMATE (sample variance / n, so it falls as the window grows
// and can therefore cross a target level), and the same relative to the pixel's
// squared mean, which is scale-free and comparable across scenes and exposures.
//
// Runs ONLY when a capture is due. Per-frame it would cost a readback wait, and
// the whole point of these numbers is to characterise a measurement window, not
// to be watched live.

#include "PassRegisters.h"

RWStructuredBuffer<float4> gVarianceM2 : BAMBOO_PASS_UAV(VARIANCE_REDUCE_REG_M2);
RWStructuredBuffer<float2> gResult     : BAMBOO_PASS_UAV(VARIANCE_REDUCE_REG_RESULT);

cbuffer VarianceReduceCB : BAMBOO_PASS_CBV(VARIANCE_REDUCE_REG_CB)
{
    uint pixelCount;
    uint sampleCount; // frames accumulated; M2 / (n-1) is the unbiased estimate
};

#define REDUCE_THREADS 256

groupshared float3 gsPartial[REDUCE_THREADS];

[numthreads(REDUCE_THREADS, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    // One group striding the whole image. A tree reduction would be faster, but
    // this runs once per capture, and being obviously correct is worth more here
    // than the microseconds a second pass would save.
    // Pixels this dark carry no signal, and dividing their (tiny, but non-zero)
    // variance by their (tinier) squared mean is how a relative metric turns into
    // a report about the background. They are counted out of both the sum and the
    // divisor, so the relative number describes the lit image.
    const float kLitLuminance = 1e-3;

    float3 local = float3(0, 0, 0); // sum variance-of-mean, sum relative, lit count
    for (uint pixel = groupIndex; pixel < pixelCount; pixel += REDUCE_THREADS)
    {
        const float4 entry = gVarianceM2[pixel];
        const float  sampleVariance = dot(entry.rgb, float3(1.0, 1.0, 1.0) / 3.0) / float(max(sampleCount - 1u, 1u));
        // What a comparison wants is the variance of the ESTIMATE, not of one
        // sample: the per-sample variance is a property of the estimator and does
        // not move as frames accumulate, so it can never cross a threshold.
        const float varianceOfMean = sampleVariance / float(max(sampleCount, 1u));

        local.x += varianceOfMean;
        if (entry.w > kLitLuminance)
        {
            local.y += varianceOfMean / (entry.w * entry.w);
            local.z += 1.0;
        }
    }

    gsPartial[groupIndex] = local;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = REDUCE_THREADS / 2; stride > 0; stride >>= 1)
    {
        if (groupIndex < stride)
            gsPartial[groupIndex] += gsPartial[groupIndex + stride];
        GroupMemoryBarrierWithGroupSync();
    }

    if (groupIndex == 0)
        gResult[0] = float2(gsPartial[0].x / float(max(pixelCount, 1u)),
                            gsPartial[0].y / max(gsPartial[0].z, 1.0));
}
