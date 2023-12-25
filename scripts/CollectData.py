from falcor import *

def render_graph_CollectData():
    g = RenderGraph("CollectData")
    CollectDataPass = createPass("CollectData")
    g.addPass(CollectDataPass, "CollectData")

    g.markOutput("CollectData.posW")
    g.markOutput("CollectData.normalW")
    g.markOutput("CollectData.wiW")
    g.markOutput("CollectData.diff")
    g.markOutput("CollectData.color")

    return g

CollectDataPass = render_graph_CollectData()
try: m.addGraph(CollectDataPass)
except NameError: None

# m.loadScene("D:/code/Falcor-Neural-Radiosity/media/test_scenes/cornell_box.pyscene")
# m.loadScene("D:/pbrt-v4-scenes/living-room/scene-exp.pbrt")
# m.loadScene("D:/pbrt-v4-scenes/pbrt-book/book.pbrt")
m.loadScene("D:/pbrt-v4-scenes/ganesha/ganesha.pbrt")
# m.loadScene("D:/code/Falcor/media/test_scenes/teapot.pyscene")

n_collect_frames = 500

for i in range(n_collect_frames):
    outputDir = "D:/code/Falcor-Neural-Radiosity/dumped_data/ganesha/frame_{:04d}".format(i)
    os.makedirs(outputDir, exist_ok=True)
    m.frameCapture.outputDir = outputDir
    render