#pragma once

#include <string>
#include <vector>
#include "magic_enum/magic_enum.hpp"

// Runtime documentation for one debug view, shown in the CVar editor
// (per-entry tooltip in the combo + text under the active selection).
struct DebugViewDoc
{
	const char* debuggedPass; // which pass / data the view inspects
	const char* expected;     // what a healthy image looks like
	const char* how;          // one-line mechanism
};

// Formats per-view docs into the strings the CVar editor displays.
// The static_assert forces every enum entry to have a doc.
template <typename EnumType, size_t N>
std::vector<std::string> FormatDebugViewDocs(const DebugViewDoc (&docs)[N])
{
	static_assert(N == magic_enum::enum_count<EnumType>(), "every debug view needs a DebugViewDoc entry");
	std::vector<std::string> formatted;
	formatted.reserve(N);
	for (const DebugViewDoc& doc : docs)
	{
		formatted.push_back(
			std::string("Debugs:   ") + doc.debuggedPass +
			"\nExpected: " + doc.expected +
			"\nHow:      " + doc.how);
	}
	return formatted;
}
