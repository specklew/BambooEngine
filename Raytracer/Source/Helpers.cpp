#include "pch.h"

#include "Helpers.h"

void ThrowIfFailed(HRESULT hr)
{
	if (FAILED(hr))
	{
		throw std::exception("Process failed with HRESULT = " + hr);
	}
}
