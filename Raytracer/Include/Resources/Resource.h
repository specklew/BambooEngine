#pragma once

class Resource
{
public:
    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12Device5> GetDevice() const { return m_device; }
    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12Resource> GetUnderlyingResource() const { return m_resource; }

    [[nodiscard]] D3D12_RESOURCE_DESC GetDesc() const
    { 
        if (!m_resource)
        {
            return m_resource->GetDesc();
        }
        return D3D12_RESOURCE_DESC{};
    }

    [[nodiscard]] bool CheckFormatSupport(D3D12_FORMAT_SUPPORT1 formatSupport1) const;
    [[nodiscard]] bool CheckFormatSupport(D3D12_FORMAT_SUPPORT2 formatSupport2) const;

    void SetResourceName(const std::wstring& name)
    {
        m_resourceName = name;
        if (m_resource)
        {
            m_resource->SetName(name.c_str());
        }
    }

    void SetResourceName(const std::string& name)
    {
        const auto w_name = std::wstring(name.begin(), name.end());
        SetResourceName(w_name);
    }
    
    [[nodiscard]] std::wstring GetResourceName() const { return m_resourceName; }
    
protected:

    Resource(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const D3D12_RESOURCE_DESC& desc, const D3D12_CLEAR_VALUE* clearValue = nullptr);
    Resource(const Microsoft::WRL::ComPtr<ID3D12Device5>& device, const Microsoft::WRL::ComPtr<ID3D12Resource> &resource, const D3D12_CLEAR_VALUE* clearValue = nullptr);

    virtual ~Resource() = default;
    
    Microsoft::WRL::ComPtr<ID3D12Device5> m_device;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_resource;
    D3D12_FEATURE_DATA_FORMAT_SUPPORT m_formatSupport;
    std::unique_ptr<D3D12_CLEAR_VALUE> m_clearValue;
    std::wstring m_resourceName;
    
    void QueryFeatureSupport();
};
