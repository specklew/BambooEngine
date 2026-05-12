#pragma once

class PlacesManager;

class SceneBookmarksPanel
{
public:
    void Draw(PlacesManager& mgr);

private:
    int  m_renamingIndex      = -1;
    int  m_pendingDeleteIndex = -1;
    char m_renameBuffer[128]  = {};
    bool m_focusRenameNext    = false;
};
