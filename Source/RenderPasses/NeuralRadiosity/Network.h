#pragma once
#include <string>
#include <cuda_runtime.h>
#include "Config.h"
#include "Params.slang"

class RadiosityNetwork
{
public:
    RadiosityNetwork(const uint32_t width, const uint32_t height);
    ~RadiosityNetwork();

    void forward(Falcor::RadiosityQuery* queries, cudaSurfaceObject_t output);

private:
    uint32_t frame_width;
    uint32_t frame_height;
};
