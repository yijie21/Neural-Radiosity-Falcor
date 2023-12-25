add_plugin(NeuralRadiosity)

target_sources(NeuralRadiosity PRIVATE
    NeuralRadiosity.cpp
    NeuralRadiosity.h
    Network.h
    Network.cu
    Tracer.rt.slang
    Config.h
    Params.slang
    ResolvePass.cs.slang
)

target_link_libraries(NeuralRadiosity PRIVATE tiny-cuda-nn)

target_copy_shaders(NeuralRadiosity RenderPasses/NeuralRadiosity)

target_source_group(NeuralRadiosity "RenderPasses")
