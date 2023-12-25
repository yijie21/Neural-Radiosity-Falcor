#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;

struct ComputeArea
{
    RWStructuredBuffer<float> areas;

    void execute(const uint3 dispatchID)
    {
        uint32_t mesh_id = dispatchID.x;
        GeometryInstanceID geoInstID = GeometryInstanceID(mesh_id, 0);
        MeshDesc meshDesc = gScene.getMeshDesc(geoInstID);
        uint32_t triangleCount = meshDesc.getTriangleCount();

        float meshArea = 0.f;
        for (uint32_t j = 0; j < triangleCount; j++)
        {
            meshArea += gScene.getFaceAreaW(geoInstID, j);
        }
        areas[mesh_id] = meshArea;
    }
}

ParameterBlock<ComputeArea> gComputeArea;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    gComputeArea.execute(dispatchThreadID);
}
