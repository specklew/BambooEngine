#pragma once

#include "Headless.h"
#include "HighResolutionClock.h"

class Renderer;

// Drives a non-interactive run: validates the requested places and techniques,
// then renders the place x technique product into a single run folder and exits.
class HeadlessRunner
{
public:
    HeadlessRunner(Renderer& renderer, HeadlessArgs args, HeadlessConfig config);

    // Returns a process exit code: 0 on success, nonzero on validation/setup failure.
    int Run();

private:
    void PumpFrame();
    bool Validate() const;

    Renderer&      m_renderer;
    HeadlessArgs   m_args;
    HeadlessConfig m_config;
    HighResolutionClock m_clock;
};
