#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

struct CollectDataParams
{
    uint2 frameDim = { 256, 256 };
    uint meshCount = 0;
    uint spp = 100;
    uint sampleCount = frameDim.x * frameDim.y;
    uint sampleWiCount = 1000;
    uint frameCount = 0;
};

END_NAMESPACE_FALCOR
