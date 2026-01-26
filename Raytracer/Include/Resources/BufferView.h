#pragma once

class Buffer;

struct BufferView
{
    std::shared_ptr<Buffer> buffer;
    UINT64 count;
    UINT64 offset;
    UINT64 offsetBytes;
    UINT64 size;
};
