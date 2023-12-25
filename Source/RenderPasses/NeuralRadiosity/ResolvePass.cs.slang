#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import RenderPasses.GBuffer.GBuffer.GBufferHelpers;

Texture2D<float4> radiance;
Texture2D<PackedHitInfo> gVBuffer;
RWTexture2D<float4> output;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadID.xy;
    const HitInfo hit = HitInfo(gVBuffer[pixel]);

    if (hit.isValid())
    {
        output[pixel] = radiance[pixel];
    }
}
