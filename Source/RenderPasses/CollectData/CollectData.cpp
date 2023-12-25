#include "CollectData.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, CollectData>();
}

using namespace pybind11::literals;

namespace
{
const char kShaderFile[] = "RenderPasses/CollectData/CollectData.rt.slang";
const char kComputeAreaFile[] = "RenderPasses/CollectData/ComputeArea.cs.slang";
const char kCAParameterBlockName[] = "gComputeArea";


// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxPayloadSizeBytes = 72u;
const uint32_t kMaxRecursionDepth = 2u;

const ChannelList kOutputChannels = {
    // clang-format off
    { "posW",           "gOutputPosWorld",   "Output world position of the first bounce", false, ResourceFormat::RGBA32Float },
    { "normalW",        "gOutputNormalWorld","Output world normal of the first bounce",   false, ResourceFormat::RGBA32Float },
    { "wiW",            "gOutputWiWorld",    "Output wi for the first bounce",            false, ResourceFormat::RGBA32Float },
    { "diff",           "gOutputDiff",       "Output diffuse",                            false, ResourceFormat::RGBA32Float },
    { "color",          "gOutputColor",      "Output color",                              false, ResourceFormat::RGBA32Float },
    // clang-format on
};

const char kComputeDirect[] = "computeDirect";
const char kUseImportanceSampling[] = "useImportanceSampling";
} // namespace

CollectData::CollectData(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    parseProperties(props);

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
}

void CollectData::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kComputeDirect)
            mComputeDirect = value;
        else if (key == kUseImportanceSampling)
            mUseImportanceSampling = value;
        else
            logWarning("Unknown property '{}' in CollectData properties.", key);
    }
}

Properties CollectData::getProperties() const
{
    Properties props;
    props[kComputeDirect] = mComputeDirect;
    props[kUseImportanceSampling] = mUseImportanceSampling;
    return props;
}

RenderPassReflection CollectData::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    const uint2 sz(mParams.frameDim.x, mParams.frameDim.y);

    // Define our input/output channels.
    addRenderPassOutputs(reflector, kOutputChannels,
        ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared | ResourceBindFlags::ShaderResource, sz);

    return reflector;
}

void CollectData::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Update refresh flag if options that affect the output have changed.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    // If we have no scene, just clear the outputs and return.
    if (!mpScene)
    {
        for (auto it : kOutputChannels)
        {
            Texture* pDst = renderData.getTexture(it.name).get();
            if (pDst)
                pRenderContext->clearTexture(pDst);
        }
        return;
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    mpCollectDataPass.pProgram->addDefines(mpSampleGenerator->getDefines());
    mpCollectDataPass.pProgram->addDefine("MAX_BOUNCES", std::to_string(10));
    mpCollectDataPass.pProgram->addDefine("COMPUTE_DIRECT", "1");
    mpCollectDataPass.pProgram->addDefine("USE_IMPORTANCE_SAMPLING", "1");
    mpCollectDataPass.pProgram->addDefine("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    mpCollectDataPass.pProgram->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    mpCollectDataPass.pProgram->addDefine("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    mpCollectDataPass.pProgram->addDefine("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");
    mpCollectDataPass.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData));
    mpCollectDataPass.pVars = RtProgramVars::create(mpDevice, mpCollectDataPass.pProgram, mpCollectDataPass.pBindingTable);

    // Compute mesh area.
    if (!mComputedMeshArea)
    {
        auto varCA = mpComputeAreaPass->getRootVar()[kCAParameterBlockName];
        varCA["areas"] = mpMeshAreaBuffer;
        mpSampleGenerator->bindShaderData(mpComputeAreaPass->getRootVar());
        mpScene->bindShaderData(mpComputeAreaPass->getRootVar()["gScene"]);

        mpComputeAreaPass->execute(pRenderContext, uint3(mpScene->getMeshCount(), 1, 1));
        mComputedMeshArea = true;
    }

    auto var = mpCollectDataPass.pVars->getRootVar();
    mpSampleGenerator->bindShaderData(var);
    var["gAreas"] = mpMeshAreaBuffer;
    var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;
    var["CB"]["gFrameCount"] = mParams.frameCount;
    var["CB"]["gMeshCount"] = mpScene->getMeshCount();
    var["CB"]["gSpp"] = mParams.spp;

    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            var[desc.texname] = renderData.getTexture(desc.name);
        }
    };
    for (auto channel : kOutputChannels)
        bind(channel);

    mpSampleGenerator->bindShaderData(mpCollectDataPass.pVars->getRootVar());
    mpScene->bindShaderData(mpCollectDataPass.pVars->getRootVar()["gScene"]);

    // Spawn the rays.
    mpScene->raytrace(pRenderContext, mpCollectDataPass.pProgram.get(), mpCollectDataPass.pVars,
                      uint3(mParams.frameDim.x, mParams.frameDim.y, 1));

    mParams.frameCount++;
}

void CollectData::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    dirty |= widget.var("Max bounces", mMaxBounces, 0u, 1u << 16);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    dirty |= widget.checkbox("Evaluate direct illumination", mComputeDirect);
    widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

    dirty |= widget.checkbox("Use importance sampling", mUseImportanceSampling);
    widget.tooltip("Use importance sampling for materials", true);

    if (dirty)
    {
        mOptionsChanged = true;
    }
}

void CollectData::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpCollectDataPass.pProgram = nullptr;
    mpCollectDataPass.pBindingTable = nullptr;
    mpCollectDataPass.pVars = nullptr;

    mpComputeAreaPass = nullptr;
    mParams.frameCount = 0;
    mComputedMeshArea = false;

    // Set new scene.
    mpScene = pScene;

    auto globalTypeConformances = mpScene->getTypeConformances();
    ProgramDesc baseDesc;
    baseDesc.addShaderModules(mpScene->getShaderModules());
    baseDesc.addTypeConformances(globalTypeConformances);

    DefineList defines;
    defines.add(mpSampleGenerator->getDefines());
    defines.add(mpScene->getSceneDefines());

    if (mpScene)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        mpCollectDataPass.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto& sbt = mpCollectDataPass.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("scatterMiss"));
        sbt->setMiss(1, desc.addMiss("shadowMiss"));

        sbt->setHitGroup(
            0,
            mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh),
            desc.addHitGroup("scatterTriangleMeshClosestHit", "scatterTriangleMeshAnyHit")
        );
        sbt->setHitGroup(
            1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowTriangleMeshAnyHit")
        );

        mpCollectDataPass.pProgram = Program::create(mpDevice, desc, defines);

        // Create compute area pass;
        ProgramDesc descArea;
        descArea.addShaderModules(mpScene->getShaderModules());
        descArea.addShaderLibrary(kComputeAreaFile).csEntry("main");
        descArea.addTypeConformances(mpScene->getTypeConformances());

        DefineList definesArea;
        definesArea.add(mpSampleGenerator->getDefines());
        definesArea.add(mpScene->getSceneDefines());

        mpComputeAreaPass = ComputePass::create(mpDevice, descArea, definesArea);

        if (!mpMeshAreaBuffer)
        {
            mpMeshAreaBuffer = mpDevice->createStructuredBuffer(sizeof(float), mpScene->getMeshCount(),
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared, MemoryType::DeviceLocal, nullptr, false);
        }
    }
}
