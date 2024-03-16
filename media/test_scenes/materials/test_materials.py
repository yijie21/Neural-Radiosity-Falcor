import glob
import os
from falcor import *

# Creates an array of default non-normal mapped materials.
def createDefaultMaterials():
    materials = []

    mtlXMtl = MxLayeredMaterial('mtlX')
    xml = """
            <materialx version="1.38" colorspace="lin_rec709" >
                <texcoord name="ST" type="vector2">
                </texcoord>
                <multiply name="noise_scale" type="vector2">
                    <input name="in1" type="vector2" nodename="ST"/>
                    <input name="in2" type="vector2" value="10, 10"/>
                </multiply>
                <oren_nayar_diffuse_bsdf name="red" type="BSDF">
                    <input name="color" type="color3" value="1, 0, 0"/>
                </oren_nayar_diffuse_bsdf>
                <image name="grid_image" type="color3">
                    <input name="texcoord" type="vector2" nodename="ST" value="0, 0" />
                    <input name="file" type="filename" value="../../test_images/grid.exr" />
                </image>
                <oren_nayar_diffuse_bsdf name="grid" type="BSDF">
                    <input name="color" type="color3" nodename="grid_image"/>
                </oren_nayar_diffuse_bsdf>
                <noise2d name="perlin" type="float">
                    <input name="texcoord" type="vector2" nodename="noise_scale"/>
                    <input name="amplitude" type="float" value="0.5"/>
                    <input name="pivot" type="float" value="0.5"/>
                </noise2d>
                <smoothstep name="switch_treshold" type="float">
                    <input name="in" type="float" nodename="perlin"/>
                    <input name="low" type="float" value="0.4"/>
                    <input name="high" type="float" value="0.6"/>
                </smoothstep>
                <mix name="mixedLobes" type="BSDF">
                    <input name="fg" type="BSDF" nodename="red" value="" />
                    <input name="bg" type="BSDF" nodename="grid" value="" />
                    <input name="mix" type="float" nodename="switch_treshold" />
                </mix>
            </materialx>
    """

    mtlXMtl.setMaterialXBuffer(xml, "mixedLobes")
    mtlXMtl.setEditableInputs(["noise_scale_in2", "switch_treshold_low", "switch_treshold_high"])
    materials.append(mtlXMtl)

    stdMtl = StandardMaterial('stdMtl')
    stdMtl.baseColor = float4(0.50, 0.70, 0.80, 1.0)
    stdMtl.roughness = 0.5
    materials.append(stdMtl)

    clothMtl = ClothMaterial('clothMtl')
    clothMtl.baseColor = float4(0.31, 0.04, 0.10, 1.0)
    clothMtl.roughness = 0.9
    materials.append(clothMtl)

    hairMtl = HairMaterial('hairMtl')
    hairMtl.baseColor = float4(1.0, 0.85, 0.45, 1.0)
    hairMtl.specularParams = float4(0.75, 0.1, 0.5, 1.0)
    materials.append(hairMtl)

    # Lambertian BRDF with kd=(0.5) stored in the MERL file format
    merlMtl = MERLMaterial('merlMtl', 'data/gray-lambert.binary')
    materials.append(merlMtl)

    return materials

# Creates an array of normal mapped materials.
def createNormalMappedMaterials(transmissive=True):
    materials = []

    stdMtl = StandardMaterial('stdMtl')
    stdMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
    stdMtl.baseColor = float4(0.50, 0.70, 0.80, 1.0)
    stdMtl.roughness = 0.5
    materials.append(stdMtl)

    clothMtl = ClothMaterial('clothMtl')
    clothMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
    clothMtl.baseColor = float4(0.31, 0.04, 0.10, 1.0)
    clothMtl.roughness = 0.9
    materials.append(clothMtl)

    searchpath = os.path.join(os.path.dirname(__file__), 'data/*.binary')
    brdfs = glob.glob(searchpath)
    merlMixMtl = MERLMixMaterial('merlMixMtl', brdfs)
    merlMixMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
    merlMixMtl.loadTexture(MaterialTextureSlot.Index, 'textures/indices.png', False)
    materials.append(merlMixMtl)

    pbrtCoatedConductorMtl = PBRTCoatedConductorMaterial('pbrtCoatedConductorMtl')
    pbrtCoatedConductorMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
    materials.append(pbrtCoatedConductorMtl)

    pbrtCoatedDiffuseMtl = PBRTCoatedDiffuseMaterial('pbrtCoatedDiffuseMtl')
    pbrtCoatedDiffuseMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
    materials.append(pbrtCoatedDiffuseMtl)

    pbrtConductorMtl = PBRTConductorMaterial('pbrtConductorMtl')
    pbrtConductorMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
    materials.append(pbrtConductorMtl)

    pbrtDiffuseMtl = PBRTDiffuseMaterial('pbrtDiffuseMtl')
    pbrtDiffuseMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
    materials.append(pbrtDiffuseMtl)

    if transmissive:
        pbrtDielectricMtl = PBRTDielectricMaterial('pbrtDielectricMtl')
        pbrtDielectricMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
        materials.append(pbrtDielectricMtl)

        pbrtDiffuseTransmissionMtl = PBRTDiffuseTransmissionMaterial('pbrtDiffuseTransmissionMtl')
        pbrtDiffuseTransmissionMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
        materials.append(pbrtDiffuseTransmissionMtl)

    return materials
