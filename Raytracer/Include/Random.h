#pragma once

namespace RaytracerRandom
{
    class Random
    {
    private:
        uint32_t m_Seed = 0;
    public:
        inline void SetSeed(uint32_t seed)
        {
            m_Seed = seed;
        }

        // Returns values 0 - 4294967295
        // source: https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
        // mentions: https://jcgt.org/published/0009/03/02/paper.pdf
        // that mentions this as the PCG source (https://github.com/imneme/pcg-cpp)
        // but the source code is so huge I genuinely cant find just the generating function lol.
        // But its on MIT/Apache 2.0
        // PCG Hash
        inline uint32_t GetRandomU32()
        {
            m_Seed = m_Seed * 747796405u + 2891336453u;
            uint32_t word = ((m_Seed >> ((m_Seed >> 28u) + 4u)) ^ m_Seed) * 277803737u;
            return (word >> 22u) ^ word;
        }

        // returns values 0 - 1 (inclusive)
        inline float GetRandomFloat()
        {
            return static_cast<float>(GetRandomU32()) / static_cast<float>(MAXUINT32);
        }
    };
        
    extern Random* g_random;
    
    // public domain, taken from http://www.isthe.com/chongo/tech/comp/fnv/index.html#FNV-source
    static constexpr uint32_t fnv1a_32(const char* buf, size_t len)
    {
        const char* bp = buf;
        const char* be = bp + len;

        uint32_t hval = 0x811c9dc5;

        while (bp < be)
        {
            hval ^= (uint32_t)*bp++;
            hval += (hval << 1) + (hval << 4) + (hval << 7) + (hval << 8) + (hval << 24);
        }

        return hval; 
    };
}