![](docs/images/teaser.png)

# Neural Radiosity (Falcor + tiny-cuda-nn)

This is an unofficial implementation of the paper [Neural Radiosity](https://saeedhd96.github.io/neural-radiosity/) on SIGGRAPH Asia 2021, with the help of the real-time rendering engine [Falcor](https://github.com/NVIDIAGameWorks/Falcor) and the lightning fast C++/CUDA neural network framework [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn). This project reproduces the neural radiosity method while running at **over 60 FPS** for radiosity inference.

Features include:
* Randomly sample the objects within a scene based on the total surface area of a triangle mesh
* Collecting training data (Position/Normal/Incident direction/Albedo/..) for **over 1300k sampled points per second** 
* Tiny-cuda-nn script for training the neural radiosity model with hashgrid
* Integration of tiny-cuda-nn into Falcor
* Customized `NeuralRadiosity` renderpass for real-time inference



## Prerequisites

1. Refer to the original repo of [Falcor](https://github.com/NVIDIAGameWorks/Falcor) to see if your computer satisfied with the minimum requirements.
2. Refer to my another repo [Falcor-tiny-cuda-nn](https://github.com/yijie21/Falcor-tiny-cuda-nn) to integrate `tiny-cuda-nn` into `Falcor`



## Preparations for Neural Radiosity

#### Create the `CollectData` renderpass:

1. Type in the terminal in the root folder of Falcor to create the pass

   ```bash
   .\tools\make_new_render_pass.bat CollectData
   ```

2. Replace the scripts in Falcor/Source/RenderPasses/CollectData with mine in this repo

#### Create the `NeuralRadiosity` renderpass:

1. Type in the terminal in the root folder of Falcor to create the pass

   ```bash
   .\tools\make_new_render_pass.bat NeuralRadiosity
   ```

2. Replace the scripts in Falcor/Source/RenderPasses/NeuralRadiosity with mine in this repo

#### Create the tiny-cuda-nn training script:

1. Create a folder in tiny-cuda-nn named `train_radiosity` and 

## Citation
If you use Falcor in a research project leading to a publication, please cite the project.
The BibTex entry is

```bibtex
@Misc{Kallweit22,
   author =      {Simon Kallweit and Petrik Clarberg and Craig Kolb and Tom{'a}{\v s} Davidovi{\v c} and Kai-Hwa Yao and Theresa Foley and Yong He and Lifan Wu and Lucy Chen and Tomas Akenine-M{\"o}ller and Chris Wyman and Cyril Crassin and Nir Benty},
   title =       {The {Falcor} Rendering Framework},
   year =        {2022},
   month =       {8},
   url =         {https://github.com/NVIDIAGameWorks/Falcor},
   note =        {\url{https://github.com/NVIDIAGameWorks/Falcor}}
}
```
