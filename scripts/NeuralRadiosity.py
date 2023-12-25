from falcor import *

def render_graph_NeuralRadiosity():
    g = RenderGraph("NeuralRadiosity")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    NeuralRadiosity = createPass("NeuralRadiosity", {'maxBounces': 3})
    g.addPass(NeuralRadiosity, "NeuralRadiosity")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addPass(VBufferRT, "VBufferRT")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("VBufferRT.vbuffer", "NeuralRadiosity.vbuffer")
    g.addEdge("VBufferRT.viewW", "NeuralRadiosity.viewW")
    g.addEdge("NeuralRadiosity.color", "AccumulatePass.input")
    g.markOutput("ToneMapper.dst")
    return g

NeuralRadiosity = render_graph_NeuralRadiosity()
try: m.addGraph(NeuralRadiosity)
except NameError: None

# m.loadScene("D:/pbrt-v4-scenes/living-room/scene-exp.pbrt")
m.loadScene("D:/code/Falcor/media/test_scenes/teapot.pyscene")
# m.loadScene("D:/pbrt-v4-scenes/ganesha/ganesha.pbrt")
