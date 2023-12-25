#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

struct CommonParams {
    uint2   frameDim = { 0, 0 };
    uint    frameCount = 0;
};

struct RadiosityQuery {
    float4 posW;
    float4 normalW;
    float4 wiW;
    float4 diff;
};

END_NAMESPACE_FALCOR
