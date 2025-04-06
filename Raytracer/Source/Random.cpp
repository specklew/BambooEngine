#include "pch.h"
#include "Random.h"

namespace RaytracerRandom
{
    static Random g_randomInstance;
    Random* g_random = &g_randomInstance;
}
