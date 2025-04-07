#pragma once


namespace nv_helpers_dx12
{
    class TopLevelASGenerator;
}

struct AccelerationStructureBuffers
{
    Microsoft::WRL::ComPtr<ID3D12Resource> p_scratch; // memory for AS builder
    Microsoft::WRL::ComPtr<ID3D12Resource> p_result; // where as is
    Microsoft::WRL::ComPtr<ID3D12Resource> p_instanceDesc; // holds the matrices of instances
};

class AccelerationStructures
{
public:

    AccelerationStructures();

    AccelerationStructureBuffers CreateBottomLevelAS(
        Microsoft::WRL::ComPtr<ID3D12Device5> device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        std::vector<std::pair<Microsoft::WRL::ComPtr<ID3D12Resource>, uint32_t>> vertexBuffers, std::vector<std::pair<
        Microsoft::WRL::ComPtr<ID3D12Resource>, uint32_t>
        > indexBuffers);
    
    void CreateTopLevelAS(
        Microsoft::WRL::ComPtr<ID3D12Device5> device,
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList,
        std::vector<std::pair<Microsoft::WRL::ComPtr<ID3D12Resource>, DirectX::XMMATRIX>>& instances, bool updateOnly);

    std::vector<std::pair<Microsoft::WRL::ComPtr<ID3D12Resource>, DirectX::XMMATRIX>>& GetInstances() { return m_instances; }
    AccelerationStructureBuffers GetTopLevelAS() { return m_topLevelASBuffers; }
    
private:
    Microsoft::WRL::ComPtr<ID3D12Resource> m_bottomLevelAS;
    std::shared_ptr<nv_helpers_dx12::TopLevelASGenerator> m_topLevelAsGenerator;
    AccelerationStructureBuffers m_topLevelASBuffers;
    std::vector<std::pair<Microsoft::WRL::ComPtr<ID3D12Resource>, DirectX::XMMATRIX>> m_instances;
    
};