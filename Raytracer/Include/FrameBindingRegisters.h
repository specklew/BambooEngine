// The one place a shader register number exists. Included by C++ and by HLSL, so the
// two cannot disagree — a mirror that has to be kept in sync by hand eventually is not
// (ADR 0019). Reflection validation still runs; it catches the failures a shared number
// cannot see (a register nothing declares, a slot nothing uses, two same-type resources
// swapped).
//
// Only numbers live here. Storage kind, heap offset and descriptor count are C++-only
// concerns and stay in the slot tables.
#ifdef __cplusplus
#pragma once
#endif

// HLSL wants the token `t7`, not the number 7, so the register letter is pasted onto the
// value. Two levels of indirection: the inner macro pastes, the outer one exists so the
// argument is expanded first (`t##n` alone would yield `tFRAME_REG_LIGHT_POOL`).
#define BAMBOO_REG_CAT2(a, b) a##b
#define BAMBOO_REG_CAT(a, b)  BAMBOO_REG_CAT2(a, b)

#define BAMBOO_CBV(n) register(BAMBOO_REG_CAT(b, n))
#define BAMBOO_SRV(n) register(BAMBOO_REG_CAT(t, n))
#define BAMBOO_UAV(n) register(BAMBOO_REG_CAT(u, n))
#define BAMBOO_SAMPLER(n) register(BAMBOO_REG_CAT(s, n))

#define BAMBOO_CBV_SPACE(n, s) register(BAMBOO_REG_CAT(b, n), BAMBOO_REG_CAT(space, s))
#define BAMBOO_SRV_SPACE(n, s) register(BAMBOO_REG_CAT(t, n), BAMBOO_REG_CAT(space, s))
#define BAMBOO_UAV_SPACE(n, s) register(BAMBOO_REG_CAT(u, n), BAMBOO_REG_CAT(space, s))

// ---------------------------------------------------------------------------
// space0 — frame layout, present in every raytracing pass whether it reads it or
// not. Keep it short; everything here is bound for every pass.
// ---------------------------------------------------------------------------

#define FRAME_MAX_TEXTURES 512

#define FRAME_REG_CAMERA_MATRICES 0
#define FRAME_REG_PASS_CONSTANTS  3

#define FRAME_REG_RAYTRACE_OUTPUT 0

#define FRAME_REG_TLAS               0
#define FRAME_REG_VERTICES           1
#define FRAME_REG_INDICES            2
#define FRAME_REG_GEOMETRY_INFO      3
#define FRAME_REG_INSTANCE_INFO      4
#define FRAME_REG_EMISSIVE_TRIANGLES 5
#define FRAME_REG_LIGHT_DATA         6
#define FRAME_REG_LIGHT_POOL         7
#define FRAME_REG_SKYBOX             8
// Last, so the bindless array can grow without moving anything above it.
#define FRAME_REG_MATERIAL_TEXTURES 16
