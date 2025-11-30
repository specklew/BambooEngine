#include "pch.h"
#include "Model.h"

#include "Resources/ConstantBuffer.h"

void Model::AddMesh(const std::shared_ptr<Primitive>& mesh)
{
    assert(mesh && "Mesh cannot be null");
    m_meshes.push_back(mesh);
}

void Model::UpdateConstantBuffer(DirectX::XMFLOAT4X4 modelWorldMatrix)
{
    m_modelWorldMatrix = modelWorldMatrix;

    DirectX::XMFLOAT4X4 dataBucket = DirectX::XMFLOAT4X4{};
    DirectX::XMFLOAT4X4* mappedData = &dataBucket;
    
    m_modelWorldMatrixBuffer->MapDataToWholeBuffer(&mappedData);
    memcpy(&mappedData[0], &modelWorldMatrix, sizeof(DirectX::XMFLOAT4X4));
    m_modelWorldMatrixBuffer->Unmap();
}
