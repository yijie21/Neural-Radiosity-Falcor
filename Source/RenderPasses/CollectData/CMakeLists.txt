add_plugin(CollectData)

target_sources(CollectData PRIVATE
    CollectData.cpp
    CollectData.h
    CollectData.rt.slang
    CollectDataParams.slang
    ComputeArea.cs.slang
)

target_copy_shaders(CollectData RenderPasses/CollectData)

target_source_group(CollectData "RenderPasses")
