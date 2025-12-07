#pragma once

#include "Resources/StringId.h"
#include "magic_enum/magic_enum.hpp"

class CVarParameter;
//struct StringId;
// Needed this enum to be visible in ProjectConfig
enum class CVarType : char
{
	Int,
	Float,
	String,
	Enum,
};
// for some reason imgui requires the bounds to be half the max values for float, otherwise throws on opening cvar floats
#define IMGUI_SAFE_FLT_MAX (FLT_MAX / 2.0f)

enum class CVarFlags : uint32_t
{
	None = 0,
	NoEdit = 1 << 1,
	EditReadOnly = 1 << 2,
	Advanced = 1 << 3,
	EditCheckbox = 1 << 4,
	EditDrag = 1 << 5,
	// space for more custom flags
};

// because the current implementation prevents templating we need to "unwrap" the enums into names and values
// this is potentially huge waste of memory (especially looking at CVarStorage struct, that has 4 instances of a type)
struct CVarEnum
{
	uint32_t index;
	std::vector<std::string> names;
	std::vector<int> values;
};

// these to enable creating composite flags in-place. e.g. CVarFlags::EditDrag | CVarFlags::Advanced in function calls
inline CVarFlags operator|(CVarFlags a, CVarFlags b) {
	using Underlying = std::underlying_type_t<CVarFlags>;
	return static_cast<CVarFlags>(static_cast<Underlying>(a) | static_cast<Underlying>(b));
}

inline CVarFlags& operator|=(CVarFlags& a, CVarFlags b) {
	a = a | b;
	return a;
}

class CVarSystem
{
public:
	static CVarSystem *Get();

	virtual CVarParameter* GetCVar(StringId hash) = 0;
	virtual bool CVarExists(StringId hash) = 0;

	virtual CVarParameter* CreateFloatCVar(const char* name, const char* description, float defaultValue, float currentVvalue, float minValue, float maxValue) = 0;
	virtual CVarParameter* CreateIntCVar(
		const char* name, const char* description, int32_t defaultValue, int32_t currentValue, int minValue, int maxValue) = 0;
	virtual CVarParameter* CreateStringCVar(
		const char* name, const char* description, const char* defaultValue, const char* current_value) = 0;
	virtual uint32_t CreateEnumCVar(const char* name, const char* description, CVarEnum value) = 0;
	
	virtual float* GetFloatCVar(StringId hash) = 0;
	virtual int32_t* GetIntCVar(StringId hash) = 0;
	virtual std::string* GetStringCVar(StringId hash) = 0;
	virtual CVarType GetCVarType(StringId hash) = 0;

	template<typename T, typename = std::enable_if_t<std::is_enum_v<T>>>
	T* GetEnumCVar(StringId hash)
	{
		auto* cvar = GetCVar(hash);
		if (cvar)
		{
			CVarEnum e = *GetEnumCVar(hash);
			return magic_enum::enum_cast<T>(e.values[e.index]);
		}
		return nullptr;
	}
	virtual CVarEnum* GetEnumCVar(StringId hash) = 0;
	// only for AutoCVarEnum::Get() internal usage
	virtual CVarEnum* GetEnumByIndex(int32_t index) = 0;

	virtual void SetCVarFloat(StringId hash, float value) = 0;
	virtual void SetCVarInt(StringId hash, int32_t value) = 0;
	virtual void SetCVarString(StringId hash, std::string&& value) = 0;
	virtual void SetCVarEnum(StringId hash, uint32_t enumValueIndex) = 0;
	// only for AutoCVarEnum::Set() internal usage
	virtual void SetCVarEnumByIndex(int32_t index, uint32_t enumValueIndex) = 0;
	virtual void DrawImguiEditor() = 0;
	virtual std::vector<std::pair<CVarType, std::string>> GetCVarParametersData() = 0;
};

template<typename T>
struct AutoCVar
{
protected:
	int m_index = 0;
	using CVarType = T;
};

// CVar for floats
// constructor syntax: (name, description, default value, optional: [ flags, minValue, maxValue ])
// using CVarFlags::EditDrag will allow you to drag the value in the imgui editor
// if using drag AND min max values are supplied, it will be displayed as a slider (with specified bounds).
// if using drag WITHOUT min max values, it will be displayed as a drag (without bounds).
struct AutoCVarFloat : AutoCVar<float>
{
	// Creates a new FLOAT cvar with the given name, description, default value and optional flags
	// example usage:
	// AutoCVarFloat cvar("player.speed", "speed", 1.0f, CVarFlags::EditDrag, 0.0f, 50.0f);
	AutoCVarFloat(const char* name, const char* description, float defaultValue, CVarFlags flags = CVarFlags::None, float minValue = -IMGUI_SAFE_FLT_MAX, float maxValue = IMGUI_SAFE_FLT_MAX);
	float Get();
	float* GetPtr();
	void Set(float value);
};

// CVar for integers
// constructor syntax: (name, description, default value, optional: [ flags, minValue, maxValue ])
// using CVarFlags::EditDrag will allow you to drag the value in the imgui editor
// if using drag AND min max values are supplied, it will be displayed as a slider (with specified bounds).
// if using drag WITHOUT min max values, it will be displayed as a drag (without bounds).
struct AutoCVarInt : AutoCVar<int32_t>
{
	// Creates a new INT cvar with the given name, description, default value, optional flags and min/max values
	// example usage:
	// AutoCVarInt cvar("player.health", "health", 100, CVarFlags::EditDrag, 0, 1000);
	AutoCVarInt(const char* name, const char* description, int32_t defaultValue, CVarFlags flags = CVarFlags::None, int minValue = INT_MIN, int maxValue = INT_MAX);
	int32_t Get();
	int32_t* GetPtr();
	void Set(int32_t val);

	void Toggle();
};

// CVar for strings
// constructor syntax: (name, description, default value, optional: [ flags ])
struct AutoCVarString : AutoCVar<std::string>
{
	// Creates a new STRING cvar with the given name, description, default value, optional flags
	AutoCVarString(const char* name, const char* description, const char* defaultValue, CVarFlags flags = CVarFlags::None);

	const char* Get();
	void Set(std::string&& val);
};

// this apparently prevents construction of this class with non-enum types
template<typename T, typename = std::enable_if_t<std::is_enum_v<T>>>
struct AutoCVarEnum : AutoCVar<T>
{
	AutoCVarEnum(const char* name, const char* description, T defaultValue, CVarFlags flags = CVarFlags::None)
	{
		CVarEnum e;

		for (auto value : magic_enum::enum_values<T>()) {
			e.names.push_back(std::string(magic_enum::enum_name(value)));
			e.values.push_back(static_cast<int>(value));
		}

		// convert default value to e.index
		uint32_t _idx = 0;
		for (auto val : magic_enum::enum_values<T>()) {
			if (val == defaultValue) {
				break;
			}
			++_idx;
		}
		e.index = _idx;
		
		uint32_t idx = CVarSystem::Get()->CreateEnumCVar(name, description, e);
		this->m_index = idx;
	}

	T Get()
	{
		CVarEnum* e = CVarSystem::Get()->GetEnumByIndex(this->m_index);
		return (magic_enum::enum_cast<T>(e->values[e->index])).value();
	}

	void Set(T value)
	{
		uint32_t idx = 0;
		for (auto val : magic_enum::enum_values<T>()) {
			if (val == value) {
				break;
			}
			++idx;
		}
		CVarSystem::Get()->SetCVarEnumByIndex(this->m_index, value);
	}
};
