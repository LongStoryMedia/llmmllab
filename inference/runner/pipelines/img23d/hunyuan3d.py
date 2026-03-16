# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from PIL import Image

from models import Model

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline


from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage


class Hunyuan3DImageTo3DPipeline:
    def __init__(self, model: Model):
        self.pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model.details.gguf_file
        )
        self.pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(
            model.details.gguf_file
        )

    def run(self, image_path: str):
        image = Image.open(image_path).convert("RGBA")
        if image.mode == "RGB":
            rembg = BackgroundRemover()
            image = rembg(image)
        mesh = self.pipeline_shapegen(image=image)[0]
        mesh = self.pipeline_texgen(mesh, image=image)
        return mesh
