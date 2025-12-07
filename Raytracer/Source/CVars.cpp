#include "pch.h"
#include "CVars.h"
#include <unordered_map>
#include <algorithm>
#include <shared_mutex>

#include "imgui.h"
#include "imgui/misc/cpp/imgui_stdlib.h"
#include "StringId.h"

class CVarParameter
{
public:
	//friend class CVarSystemImpl;

	int32_t arrayIndex;

	CVarType type;
	CVarFlags flags;
	std::string name;
	std::string description;
};

struct CVarEnumStorage
{
	CVarEnum value;
	uint32_t initial;
	uint32_t min;
	uint32_t max;
	CVarParameter* parameter;
};

template<typename T>
struct CVarStorage
{
	T initial;
	T current;
	T min;
	T max;
	// TODO : add step
	CVarParameter* parameter;
};

template<typename T>
struct CVarArray
{
	CVarStorage<T>* cvars;
	int32_t lastCVar{ 0 };

	CVarArray(size_t size)
	{
		cvars = new CVarStorage<T>[size]();
	}
	~CVarArray()
	{
		delete[] cvars;
	}

	T GetCurrent(int32_t index)
	{
		return cvars[index].current;
	};

	T *GetCurrentPtr(int32_t index) 
	{
		return &cvars[index].current;
	}

	T GetMin(int32_t index)
	{
		return cvars[index].min;
	}

	T GetMax(int32_t index)
	{
		return cvars[index].max;
	}

	void SetCurrent(const T& val, int32_t index)
	{
		cvars[index].current = val;
	}

	int Add(const T& value, CVarParameter* param)
	{
		int index = lastCVar;

		cvars[index].current = value;
		cvars[index].initial = value;
		cvars[index].parameter = param;

		param->arrayIndex = index;
		lastCVar++;
		return index;
	}

	int Add(const T& initialValue, const T& currentValue, const T& minValue, const T& maxValue, CVarParameter* param)
	{
		int index = lastCVar;

		cvars[index].current = currentValue;
		cvars[index].initial = initialValue;
		cvars[index].parameter = param;
		cvars[index].min = minValue;
		cvars[index].max = maxValue;

		param->arrayIndex = index;
		lastCVar++;

		return index;
	}
};

struct CVarEnumArray
{
	CVarEnumStorage* cvarEnums;
	int32_t lastCVar{ 0 };

	CVarEnumArray(size_t size)
	{
		cvarEnums = new CVarEnumStorage[size]();
	}
	~CVarEnumArray()
	{
		delete[] cvarEnums;
	}

	int GetCurrent(int32_t index)
	{
		auto& cvar = cvarEnums[index];
		return cvar.value.values[cvar.value.index];
	};

	CVarEnum* GetCurrentEnumPtr(int32_t index)
	{
		return &cvarEnums[index].value;
	}

	uint32_t GetMin(int32_t index)
	{
		return cvarEnums[index].min;
	}

	uint32_t GetMax(int32_t index)
	{
		return cvarEnums[index].max;
	}

	void SetCurrent(const uint32_t val, int32_t index)
	{
		cvarEnums[index].value.index = val;
	}

	int Add(const CVarEnum& value, CVarParameter* param)
	{
		int index = lastCVar;

		cvarEnums[index].value = value;
		cvarEnums[index].initial = value.index;
		cvarEnums[index].parameter = param;

		param->arrayIndex = index;
		lastCVar++;
		return index;
	}
};

class CVarSystemImpl : public CVarSystem
{
public:
	CVarParameter* GetCVar(StringId hash) override final;
	bool CVarExists(StringId hash) override final;
	
	CVarParameter* CreateFloatCVar(const char* name, const char* description, float defaultValue, float currentValue, float minValue, float maxValue) override final;
	CVarParameter* CreateIntCVar(const char* name, const char* description, int32_t defaultValue, int32_t currentValue, int minValue, int maxValue) override final;
	CVarParameter* CreateStringCVar(const char* name, const char* description, const char* defaultValue, const char* currentValue) override final;
	uint32_t CreateEnumCVar(const char* name, const char* description, CVarEnum value) override final;
	
	float* GetFloatCVar(StringId hash) override final;
	int32_t* GetIntCVar(StringId hash) override final;
	std::string* GetStringCVar(StringId hash) override final;
	CVarEnum* GetEnumCVar(StringId hash) override final;
	CVarEnum* GetEnumByIndex(int32_t index) override final;
	
	void SetCVarFloat(StringId hash, float value) override final;
	void SetCVarInt(StringId hash, int32_t value) override final;
	void SetCVarString(StringId hash, std::string&& value) override final;
	void SetCVarEnum(StringId hash, uint32_t enumValueIndex) override final;
	void SetCVarEnumByIndex(int32_t index, uint32_t enumValueIndex) override final;

	CVarType GetCVarType(StringId hash) override final;

	void DrawImguiEditor() override final;
	std::vector<std::pair<CVarType, std::string>> GetCVarParametersData() final;
	void EditParameter(CVarParameter* p, float textWidth);

	constexpr static int MAX_INT_CVARS = 1000;
	CVarArray<int32_t> intCVars2{ MAX_INT_CVARS };

	constexpr static int MAX_FLOAT_CVARS = 1000;
	CVarArray<float> floatCVars{ MAX_FLOAT_CVARS };

	constexpr static int MAX_STRING_CVARS = 200;
	CVarArray<std::string> stringCVars{ MAX_STRING_CVARS };

	constexpr static int MAX_ENUM_CVARS = 100;
	CVarEnumArray enumCVars{ MAX_ENUM_CVARS };

	//using templates with specializations to get the cvar arrays for each type.
	//if you try to use a type that doesnt have specialization, it will trigger a linker error
	template<typename T>
	CVarArray<T>* GetCVarArray();

	template<>
	CVarArray<int32_t>* GetCVarArray()
	{
		return &intCVars2;
	}
	template<>
	CVarArray<float>* GetCVarArray()
	{
		return &floatCVars;
	}
	template<>
	CVarArray<std::string>* GetCVarArray()
	{
		return &stringCVars;
	}

	//templated get-set cvar versions for syntax sugar
	template <typename T>
	T* GetCVarCurrent(StringId namehash)
	{
		CVarParameter* par = GetCVar(namehash);
		if (!par)
		{
			return nullptr;
		}
		else
		{
			return GetCVarArray<T>()->GetCurrentPtr(par->arrayIndex);
		}
	}
	
	template <>
	CVarEnum* GetCVarCurrent<CVarEnum>(StringId namehash)
	{
		CVarParameter* par = GetCVar(namehash);
		if (!par || par->type != CVarType::Enum)
		{
			return nullptr;
		}
		return enumCVars.GetCurrentEnumPtr(par->arrayIndex);
	}

	template<typename T>
	void SetCVarCurrent(StringId namehash, const T& value)
	{
		CVarParameter* cvar = GetCVar(namehash);
		if (cvar)
		{
			GetCVarArray<T>()->SetCurrent(value, cvar->arrayIndex);
		}
	}

	template<>
	void SetCVarCurrent<uint32_t>(StringId namehash, const uint32_t& value)
	{
		CVarParameter* cvar = GetCVar(namehash);
		if (cvar && cvar->type == CVarType::Enum)
		{
			enumCVars.SetCurrent(value, cvar->arrayIndex);
		}
	}

	static CVarSystemImpl* Get()
	{
		return static_cast<CVarSystemImpl*>(CVarSystem::Get());
	}

private:
	static std::string StripCVarName(const std::string& fullName);
	CVarParameter* InitCVar(const char* name, const char* description);

	std::shared_mutex m_mutex;
	std::unordered_map<StringId, CVarParameter> m_savedCVars;
	std::vector<CVarParameter*> m_cachedEditParameters;
};

CVarSystem* CVarSystem::Get()
{
	static CVarSystemImpl cvarSys{};
	return &cvarSys;
}

CVarParameter* CVarSystemImpl::GetCVar(StringId hash)
{
	std::shared_lock lock(m_mutex);
	auto it = m_savedCVars.find(hash);

	if (it != m_savedCVars.end())
	{
		return &(it->second);
	}

	assert(false && "Cvar not found!");

	return nullptr;
}

bool CVarSystemImpl::CVarExists(StringId hash)
{
	// this exists because of asserts eh, is only used for int/enum checks in project config
	std::shared_lock lock(m_mutex);
	auto it = m_savedCVars.find(hash);

	if (it != m_savedCVars.end())
	{
		return true;
	}

	return false;
}

AutoCVarFloat::AutoCVarFloat(const char* name, const char* description, float defaultValue, CVarFlags flags, float minValue, float maxValue)
{
	CVarParameter* cvar = CVarSystem::Get()->CreateFloatCVar(name, description, defaultValue, defaultValue, minValue, maxValue);
	cvar->flags = flags;
	m_index = cvar->arrayIndex;
}

template <typename T>
T GetCVarCurrentByIndex(int32_t index)
{
	return CVarSystemImpl::Get()->GetCVarArray<T>()->GetCurrent(index);
}

template <typename T>
T* PtrGetCVarCurrentByIndex(int32_t index)
{
	return CVarSystemImpl::Get()->GetCVarArray<T>()->GetCurrentPtr(index);
}

template <typename T>
void SetCVarCurrentByIndex(int32_t index, const T& data)
{
	CVarSystemImpl::Get()->GetCVarArray<T>()->SetCurrent(data, index);
}

float AutoCVarFloat::Get()
{
	return GetCVarCurrentByIndex<CVarType>(m_index);
}

float* AutoCVarFloat::GetPtr()
{
	return PtrGetCVarCurrentByIndex<CVarType>(m_index);
}

void AutoCVarFloat::Set(float value)
{
	SetCVarCurrentByIndex<CVarType>(m_index, value);
}

AutoCVarInt::AutoCVarInt(const char* name, const char* description, int32_t defaultValue, CVarFlags flags, int minValue, int maxValue)
{
	CVarParameter* cvar = CVarSystem::Get()->CreateIntCVar(name, description, defaultValue, defaultValue, minValue, maxValue);
	cvar->flags = flags;
	m_index = cvar->arrayIndex;
}

int32_t AutoCVarInt::Get()
{
	return GetCVarCurrentByIndex<CVarType>(m_index);
}

int32_t* AutoCVarInt::GetPtr()
{
	return PtrGetCVarCurrentByIndex<CVarType>(m_index);
}

void AutoCVarInt::Set(int32_t val)
{
	SetCVarCurrentByIndex<CVarType>(m_index, val);
}

void AutoCVarInt::Toggle()
{
	bool enabled = Get() != 0;

	Set(enabled ? 0 : 1);
}

AutoCVarString::AutoCVarString(const char* name, const char* description, const char* defaultValue, CVarFlags flags)
{
	CVarParameter* cvar = CVarSystem::Get()->CreateStringCVar(name, description, defaultValue, defaultValue);
	cvar->flags = flags;
	m_index = cvar->arrayIndex;
}

const char* AutoCVarString::Get()
{
	return GetCVarCurrentByIndex<CVarType>(m_index).c_str();
};

void AutoCVarString::Set(std::string&& val)
{
	SetCVarCurrentByIndex<CVarType>(m_index, val);
}

CVarParameter* CVarSystemImpl::CreateFloatCVar(const char* name, const char* description, float defaultValue, float currentValue, float minValue, float maxValue)
{
	std::unique_lock lock(m_mutex);
	CVarParameter* param = InitCVar(name, description);
	if (!param) return nullptr;

	param->type = CVarType::Float;

	GetCVarArray<float>()->Add(defaultValue, currentValue, minValue, maxValue, param);

	return param;
}

CVarParameter* CVarSystemImpl::CreateIntCVar(const char* name, const char* description, int32_t defaultValue, int32_t currentValue, int32_t minValue, int32_t maxValue)
{
	std::unique_lock lock(m_mutex);
	CVarParameter* param = InitCVar(name, description);
	if (!param) return nullptr;

	param->type = CVarType::Int;

	GetCVarArray<int32_t>()->Add(defaultValue, currentValue, minValue, maxValue, param);

	return param;
}

CVarParameter* CVarSystemImpl::CreateStringCVar(const char* name, const char* description, const char* defaultValue, const char* currentValue)
{
	std::unique_lock lock(m_mutex);
	CVarParameter* param = InitCVar(name, description);
	if (!param) return nullptr;

	param->type = CVarType::String;

	// min and max values are not applicable to string, but valueas are still needed so we can use random strings
	GetCVarArray<std::string>()->Add(defaultValue, currentValue, "a", "a", param);

	return param;
}

uint32_t CVarSystemImpl::CreateEnumCVar(const char* name, const char* description, CVarEnum value)
{
	CVarParameter* param = InitCVar(name, description);
	assert(param != nullptr && "creation of Enum cvar failed.");

	param->type = CVarType::Enum;

	enumCVars.Add(value, param);

	return param->arrayIndex;
}

float* CVarSystemImpl::GetFloatCVar(StringId hash)
{
	return GetCVarCurrent<float>(hash);
}

int32_t* CVarSystemImpl::GetIntCVar(StringId hash)
{
	return GetCVarCurrent<int32_t>(hash);
}

std::string* CVarSystemImpl::GetStringCVar(StringId hash)
{
	return GetCVarCurrent<std::string>(hash);
}

CVarEnum* CVarSystemImpl::GetEnumCVar(StringId hash)
{
	return GetCVarCurrent<CVarEnum>(hash);
}

CVarEnum* CVarSystemImpl::GetEnumByIndex(int32_t index)
{
	return enumCVars.GetCurrentEnumPtr(index);
}

void CVarSystemImpl::SetCVarFloat(StringId hash, float value)
{
	SetCVarCurrent<float>(hash, value);
}

void CVarSystemImpl::SetCVarInt(StringId hash, int32_t value)
{
	SetCVarCurrent<int32_t>(hash, value);
}

void CVarSystemImpl::SetCVarString(StringId hash, std::string&& value)
{
	SetCVarCurrent<std::string>(hash, value);
}

void CVarSystemImpl::SetCVarEnum(StringId hash, uint32_t enumValueIndex)
{
	SetCVarCurrent<uint32_t>(hash, enumValueIndex);
}

void CVarSystemImpl::SetCVarEnumByIndex(int32_t index, uint32_t enumValueIndex)
{
	enumCVars.SetCurrent(enumValueIndex, index);
}

CVarType CVarSystemImpl::GetCVarType(StringId hash)
{
	return m_savedCVars[hash].type;
}

CVarParameter* CVarSystemImpl::InitCVar(const char* name, const char* description)
{
	const StringId nameHash = StringId(name);
	m_savedCVars[nameHash] = CVarParameter{};

	CVarParameter& newParam = m_savedCVars[nameHash];

	newParam.name = name;
	newParam.description = description;

	return &newParam;
}

void CVarSystemImpl::DrawImguiEditor()
{
	static std::string searchText = "";

	ImGui::Begin("CVar Editor");

	const float filterWidth = ImGui::GetContentRegionAvail().x;
	static ImGuiTextFilter filter;
	filter.Draw("Search", filterWidth * 0.85f);
	static bool bShowAdvanced = false;
	ImGui::Checkbox("Advanced", &bShowAdvanced);
	ImGui::SameLine(0.0, 30.0);
	ImGui::Text("Sliders: CTRL+CLICK to input value.");

	ImGui::Separator();
	m_cachedEditParameters.clear();

	auto addToEditList = [&](auto parameter)
	{
		const bool bHidden = ((uint32_t)parameter->flags & (uint32_t)CVarFlags::NoEdit);
		const bool bIsAdvanced = ((uint32_t)parameter->flags & (uint32_t)CVarFlags::Advanced);

		if (!bHidden)
		{
			if (!(!bShowAdvanced && bIsAdvanced) && filter.PassFilter(parameter->name.c_str()))
			{
				m_cachedEditParameters.push_back(parameter);
			};
		}
	};

	for (int i = 0; i < GetCVarArray<int32_t>()->lastCVar; i++)
	{
		addToEditList(GetCVarArray<int32_t>()->cvars[i].parameter);
	}
	for (int i = 0; i < GetCVarArray<float>()->lastCVar; i++)
	{
		addToEditList(GetCVarArray<float>()->cvars[i].parameter);
	}
	for (int i = 0; i < GetCVarArray<std::string>()->lastCVar; i++)
	{
		addToEditList(GetCVarArray<std::string>()->cvars[i].parameter);
	}
	for (int i = 0; i < enumCVars.lastCVar; i++)
	{
		addToEditList(enumCVars.cvarEnums[i].parameter);
	}

	if (m_cachedEditParameters.size() > 3)
	{
		std::unordered_map<std::string, std::unordered_map<std::string, std::vector<CVarParameter*>>> categorizedParams;

		// Insert all the edit parameters into the hashmap by category and subcategory
		for (auto p : m_cachedEditParameters)
		{
			size_t firstDotPos = p->name.find('.');
			size_t secondDotPos = firstDotPos != std::string::npos ? p->name.find('.', firstDotPos + 1) : -1;

			std::string category = firstDotPos != -1 ? p->name.substr(0, firstDotPos) : "Uncategorized";
			std::string subcategory = (secondDotPos != -1) ? p->name.substr(firstDotPos + 1, secondDotPos - firstDotPos - 1) : "";

			categorizedParams[category][subcategory].push_back(p);
		}

		for (auto& [category, subcategories] : categorizedParams)
		{
			if (ImGui::CollapsingHeader(category.c_str()))
			{
				for (auto& [subcategory, parameters] : subcategories)
				{
					// case insensitive alphabetical sort of parameters
					std::sort(parameters.begin(), parameters.end(), [](CVarParameter* A, CVarParameter* B)
					{
						const std::string& nameA = A->name;
						const std::string& nameB = B->name;

						return std::lexicographical_compare(
							nameA.begin(), nameA.end(),
							nameB.begin(), nameB.end(),
							[](char a, char b) {
								return std::tolower(static_cast<unsigned char>(a)) < std::tolower(static_cast<unsigned char>(b));
							}
						);
					});

					if (!subcategory.empty())
					{
						if (ImGui::TreeNode(subcategory.c_str()))
						{
							float maxTextWidth = 0;

							for (auto p : parameters)
							{
								maxTextWidth = std::max(maxTextWidth, ImGui::CalcTextSize(p->name.c_str()).x);
							}
							for (auto p : parameters)
							{
								EditParameter(p, maxTextWidth);
							}

							ImGui::TreePop();
						}
					}
					else
					{
						float maxTextWidth = 0;

						for (auto p : parameters)
						{
							maxTextWidth = std::max(maxTextWidth, ImGui::CalcTextSize(p->name.c_str()).x);
						}
						for (auto p : parameters)
						{
							EditParameter(p, maxTextWidth);
						}
					}
				}
			}
		}
	}
	else
	{
		// Alphabetical sort
		std::sort(m_cachedEditParameters.begin(), m_cachedEditParameters.end(), [](CVarParameter* A, CVarParameter* B)
			{
				return A->name < B->name;
			});

		float maxTextWidth = 0;
		for (auto p : m_cachedEditParameters)
		{
			maxTextWidth = std::max(maxTextWidth, ImGui::CalcTextSize(p->name.c_str()).x);
		}
		for (auto p : m_cachedEditParameters)
		{
			EditParameter(p, maxTextWidth);
		}
	}

	ImGui::End();
}

std::vector<std::pair<CVarType, std::string>> CVarSystemImpl::GetCVarParametersData()
{
	std::vector<std::pair<CVarType, std::string>> parameters;
	parameters.reserve(m_savedCVars.size());
	for (auto &[key, parameter] : m_savedCVars)
	{
		parameters.push_back({ parameter.type, parameter.name });
	}

	return std::move(parameters);
}

static void Label(const char* label, float textWidth)
{
#ifndef SHIPPING
	constexpr float slack = 50;
	constexpr float editorWidth = 100;

	ImGui::Text(label);

	ImGui::SameLine();
	const ImVec2 startPos = ImGui::GetCursorScreenPos();
	const float fullWidth = textWidth + slack;
	const ImVec2 finalPos = { startPos.x + fullWidth, startPos.y };
	ImGui::SetCursorScreenPos(finalPos);

	ImGui::SetNextItemWidth(editorWidth);
#endif
}

void CVarSystemImpl::EditParameter(CVarParameter* p, float textWidth)
{
#ifndef SHIPPING
    const bool readonlyFlag = ((uint32_t)p->flags & (uint32_t)CVarFlags::EditReadOnly);
    const bool checkboxFlag = ((uint32_t)p->flags & (uint32_t)CVarFlags::EditCheckbox);
    const bool dragFlag = ((uint32_t)p->flags & (uint32_t)CVarFlags::EditDrag);
    const std::string cVarName = StripCVarName(p->name);

    // Reduce vertical padding
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(2.0f, 0.0f)); // Adjust horizontal and vertical padding

    if (ImGui::BeginTable("CVarTable", 2, ImGuiTableFlags_SizingStretchSame))
    {
        // First column: Parameter name
        ImGui::TableNextColumn();
        ImGui::Text("%s", cVarName.c_str());

        // Second column: Editable field
        ImGui::TableNextColumn();
        switch (p->type)
        {
        case CVarType::Int:
            if (readonlyFlag)
            {
                ImGui::Text("%i", GetCVarArray<int32_t>()->GetCurrent(p->arrayIndex));
            }
            else
            {
                ImGui::PushID(p->name.c_str());
                if (checkboxFlag)
                {
                    bool bCheckbox = GetCVarArray<int32_t>()->GetCurrent(p->arrayIndex) != 0;
                    if (ImGui::Checkbox("", &bCheckbox))
                    {
                        GetCVarArray<int32_t>()->SetCurrent(bCheckbox ? 1 : 0, p->arrayIndex);
                    }
                }
                else if (dragFlag)
                {
                    const int32_t min = GetCVarArray<int32_t>()->GetMin(p->arrayIndex);
                    const int32_t max = GetCVarArray<int32_t>()->GetMax(p->arrayIndex);
                    if (min == INT_MIN && max == INT_MAX)
                    {
                        ImGui::DragInt("", GetCVarArray<int32_t>()->GetCurrentPtr(p->arrayIndex), 1.0f, min, max);
                    }
                    else
                    {
                        ImGui::SliderInt("", GetCVarArray<int32_t>()->GetCurrentPtr(p->arrayIndex), min, max);
                    }
                }
                else
                {
                    ImGui::InputInt("", GetCVarArray<int32_t>()->GetCurrentPtr(p->arrayIndex));
                }

                ImGui::SameLine();
                if (ImGui::Button("R"))
                {
                    GetCVarArray<int32_t>()->SetCurrent(GetCVarArray<int32_t>()->cvars[p->arrayIndex].initial, p->arrayIndex);
                }
                ImGui::PopID();
            }
            break;

        case CVarType::Float:
            if (readonlyFlag)
            {
                ImGui::Text("%.3f", GetCVarArray<float>()->GetCurrent(p->arrayIndex));
            }
            else
            {
                ImGui::PushID(p->name.c_str());
                if (dragFlag)
                {
                    float min = GetCVarArray<float>()->GetMin(p->arrayIndex);
                    float max = GetCVarArray<float>()->GetMax(p->arrayIndex);
                    if (min == -IMGUI_SAFE_FLT_MAX && max == IMGUI_SAFE_FLT_MAX)
                    {
                        ImGui::DragFloat("", GetCVarArray<float>()->GetCurrentPtr(p->arrayIndex), 0.05f, min, max);
                    }
                    else
                    {
                        ImGui::SliderFloat("", GetCVarArray<float>()->GetCurrentPtr(p->arrayIndex), min, max, "%.3f");
                    }
                }
                else
                {
                    ImGui::InputFloat("", GetCVarArray<float>()->GetCurrentPtr(p->arrayIndex), 0, 0, "%.3f");
                }

                ImGui::SameLine();
                if (ImGui::Button("R"))
                {
                    GetCVarArray<float>()->SetCurrent(GetCVarArray<float>()->cvars[p->arrayIndex].initial, p->arrayIndex);
                }
                ImGui::PopID();
            }
            break;

        case CVarType::String:
            if (readonlyFlag)
            {
                ImGui::Text("%s", GetCVarArray<std::string>()->GetCurrent(p->arrayIndex).c_str());
            }
            else
            {
                ImGui::PushID(p->name.c_str());
                ImGui::InputText("", GetCVarArray<std::string>()->GetCurrentPtr(p->arrayIndex));
                ImGui::SameLine();
                if (ImGui::Button("R"))
                {
                    GetCVarArray<std::string>()->SetCurrent(GetCVarArray<std::string>()->cvars[p->arrayIndex].initial, p->arrayIndex);
                }
                ImGui::PopID();
            }
            break;
        case CVarType::Enum:
	        {
		        const StringId hash = StringId(p->name);
        		CVarEnum* enumCVar = CVarSystem::Get()->GetEnumCVar(hash);
				
        		if (!enumCVar || enumCVar->names.empty()) 
        			return;

        		// Ensure index is within bounds
        		int selectedIndex = std::clamp(static_cast<int>(enumCVar->index), 0, static_cast<int>(enumCVar->values.size() - 1));
        		// Render ImGui combo box
        		ImGui::PushID(p->name.c_str());
        		if (ImGui::Combo(("##" + p->name).c_str(), &selectedIndex, 
								 [](void* data, int idx, const char** out_text) {
									 auto* names = static_cast<std::vector<std::string>*>(data);
									 if (out_text) *out_text = (*names)[idx].c_str();
									 return true;
								 }, 
								 static_cast<void*>(&enumCVar->names), 
								 static_cast<int>(enumCVar->names.size())))
        		{
        			CVarSystem::Get()->SetCVarEnum(hash, selectedIndex);
        		}
        		ImGui::SameLine();
        		if (ImGui::Button("R"))
        		{
        			//GetCVarArray<std::string>()->SetCurrent(GetCVarArray<std::string>()->cvars[p->arrayIndex].initial, p->arrayIndex);
        			enumCVars.SetCurrent(enumCVars.cvarEnums[p->arrayIndex].initial, p->arrayIndex);
        		}
        		ImGui::PopID();
        		break;
	        }
        default:
            break;
        }

        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("%s", p->description.c_str());
        }

        ImGui::EndTable();
    }

    // Restore the previous style
    ImGui::PopStyleVar();
#endif
}

std::string CVarSystemImpl::StripCVarName(const std::string& fullName)
{
	const size_t lastDotPos = fullName.rfind('.');

	if (lastDotPos == std::string::npos)
	{
		return fullName;
	}
	return fullName.substr(lastDotPos + 1);
}
