#pragma once

#include "Headless.h"
#include "HighResolutionClock.h"
#include "RenderTechnique.h"
#include "ScreenshotManager.h"

class Renderer;

// Drives a non-interactive run: validates the requested places and techniques,
// then renders the place x technique x debug-view product into a single run
// folder and exits.
class HeadlessRunner
{
public:
    HeadlessRunner(Renderer& renderer, HeadlessArgs args, HeadlessConfig config);

    // Returns a process exit code: 0 on success, nonzero on validation/setup failure.
    int Run();

private:
    void PumpFrame();
    void ApplyConfiguredLights();
    bool Validate() const;
    std::vector<RenderTechnique::DebugView> ResolveDebugViews() const;

    // Renders the real workload and throws the result away until the frame time
    // stops moving. Returns the seconds actually spent, which goes into the
    // sidecar — a warm-up nobody can see the length of is not a protocol.
    float WarmUp();

    // The capture budget plus the checkpoints inside it, resolved from the flags.
    CaptureSchedule BuildSchedule() const;

    Renderer&      m_renderer;
    HeadlessArgs   m_args;
    HeadlessConfig m_config;
    HighResolutionClock m_clock;
};
