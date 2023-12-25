#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "CollectDataParams.slang"

using namespace Falcor;

class CollectData : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(CollectData, "CollectData", "Collect Data Using Minimal Pathtracer.");

    static ref<CollectData> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<CollectData>(pDevice, props);
    }

    CollectData(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    static void registerScriptBindings(pybind11::module& m);

private:
    void parseProperties(const Properties& props);

    // Internal state

    /// Current scene.
    ref<Scene> mpScene;
    /// GPU sample generator.
    ref<SampleGenerator> mpSampleGenerator;

    // Parameters shared with the shaders
    CollectDataParams mParams;

    // Configuration

    /// Max number of indirect bounces (0 = none).
    uint mMaxBounces = 3;
    /// Compute direct illumination (otherwise indirect only).
    bool mComputeDirect = true;
    /// Use importance sampling for materials.
    bool mUseImportanceSampling = true;

    // Runtime data

    /// Frame count since scene was loaded.
    bool mOptionsChanged = false;

    struct
    {
        ref<Program> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;
    } mpCollectDataPass;

    bool mComputedMeshArea = false;
    ref<Buffer> mpMeshAreaBuffer;
    ref<ComputePass> mpComputeAreaPass;
};
