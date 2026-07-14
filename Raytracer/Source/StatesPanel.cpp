#include "pch.h"
#include "StatesPanel.h"

#include "StatesManager.h"

#include "imgui.h"

void StatesPanel::Draw(StatesManager& mgr)
{
    ImGui::Begin("States");

    if (mgr.HasCurrentScene())
        ImGui::Text("Scene: %s", mgr.GetCurrentScene().c_str());
    else
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "No scene loaded");

    ImGui::Separator();

    const std::vector<State> states = mgr.GetStatesForCurrentScene();
    for (size_t i = 0; i < states.size(); ++i)
    {
        ImGui::PushID(static_cast<int>(i));

        const bool isRenaming = (m_renamingIndex == static_cast<int>(i));
        if (isRenaming)
        {
            ImGui::SetNextItemWidth(180.0f);
            if (m_focusRenameNext)
            {
                ImGui::SetKeyboardFocusHere();
                m_focusRenameNext = false;
            }

            const bool committed = ImGui::InputText(
                "##rename",
                m_renameBuffer,
                sizeof(m_renameBuffer),
                ImGuiInputTextFlags_EnterReturnsTrue);

            const bool deactivatedAfterEdit = ImGui::IsItemDeactivatedAfterEdit();
            const bool deactivatedNoEdit    = ImGui::IsItemDeactivated() && !deactivatedAfterEdit;

            if (committed || deactivatedAfterEdit)
            {
                mgr.RenameState(static_cast<size_t>(m_renamingIndex), m_renameBuffer);
                m_renamingIndex = -1;
            }
            else if (deactivatedNoEdit)
            {
                m_renamingIndex = -1;
            }
        }
        else
        {
            ImGui::TextUnformatted(states[i].name.c_str());
            ImGui::SameLine();
            if (states[i].hasLights)
                ImGui::TextDisabled("(%zu lights)", states[i].lights.size());
            else
                ImGui::TextDisabled("(cam only)");
        }

        ImGui::SameLine();
        if (ImGui::Button("Go To"))
        {
            m_renamingIndex = -1;
            mgr.GoToState(i);
        }

        ImGui::SameLine();
        if (ImGui::Button("Rename"))
        {
            m_renamingIndex = static_cast<int>(i);
            strncpy_s(m_renameBuffer, states[i].name.c_str(), _TRUNCATE);
            m_focusRenameNext = true;
        }

        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.1f, 0.1f, 1.0f));
        if (ImGui::Button("Delete"))
            m_pendingDeleteIndex = static_cast<int>(i);
        ImGui::PopStyleColor();

        ImGui::PopID();
    }

    ImGui::Separator();

    ImGui::BeginDisabled(!mgr.HasCurrentScene());
    if (ImGui::Button("+ New State"))
    {
        if (mgr.AddStateFromScene())
        {
            const std::vector<State> refreshed = mgr.GetStatesForCurrentScene();
            if (!refreshed.empty())
            {
                m_renamingIndex = static_cast<int>(refreshed.size() - 1);
                strncpy_s(m_renameBuffer, refreshed.back().name.c_str(), _TRUNCATE);
                m_focusRenameNext = true;
            }
        }
    }
    ImGui::EndDisabled();

    if (!mgr.GetActiveStateName().empty())
    {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f), "Currently at: %s", mgr.GetActiveStateName().c_str());
    }

    if (m_pendingDeleteIndex >= 0)
        ImGui::OpenPopup("Confirm Delete");

    if (ImGui::BeginPopupModal("Confirm Delete", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        if (m_pendingDeleteIndex >= 0 && static_cast<size_t>(m_pendingDeleteIndex) < states.size())
            ImGui::Text("Delete '%s'?", states[m_pendingDeleteIndex].name.c_str());
        else
            ImGui::Text("Delete?");

        if (ImGui::Button("Delete", ImVec2(120, 0)))
        {
            if (m_pendingDeleteIndex >= 0)
                mgr.DeleteState(static_cast<size_t>(m_pendingDeleteIndex));
            m_pendingDeleteIndex = -1;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0)))
        {
            m_pendingDeleteIndex = -1;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    ImGui::End();
}
