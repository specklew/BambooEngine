#include "pch.h"
#include "RenderTechnique.h"

#include "BufferDebugView.h"

std::vector<RenderTechnique::Entry>& RenderTechnique::GetRegistry()
{
    static std::vector<Entry> registry;
    return registry;
}

int RenderTechnique::RegisterTechnique(std::string name, std::function<std::shared_ptr<RenderTechnique>()> factory)
{
    GetRegistry().push_back({std::move(name), std::move(factory)});
    return static_cast<int>(GetRegistry().size()) - 1;
}

std::vector<RenderTechnique::DebugView> RenderTechnique::WithBufferViews(std::vector<DebugView> own)
{
    // "None" is already the technique's first entry, so the shared list drops its
    // own — one "off" in the menu, not two.
    for (DebugView& view : BuildDebugViews<BufferDebugView>(kBufferDebugViewDocs))
    {
        if (view.index == static_cast<int>(BufferDebugView::None))
            continue;
        view.index += kBufferViewIndexBase;
        own.push_back(std::move(view));
    }
    return own;
}

bool RenderTechnique::SelectDebugView(RenderTechnique& technique, int index)
{
    if (index >= kBufferViewIndexBase)
    {
        const auto view = magic_enum::enum_cast<BufferDebugView>(index - kBufferViewIndexBase);
        if (!view)
            return false;
        // A buffer view replaces the frame, so the technique's own view goes off.
        technique.SetDebugView(0);
        g_bufferDebugView.Set(*view);
        return true;
    }

    g_bufferDebugView.Set(BufferDebugView::None);
    return technique.SetDebugView(index);
}
