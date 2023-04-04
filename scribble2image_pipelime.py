import random
import typing as t

import cv2
import einops
import numpy as np
import pipelime.items as pli
import pipelime.sequences as pls
import pipelime.stages as plst
import pydantic as pyd
import torch
from pytorch_lightning import seed_everything

import config as config
from annotator.util import HWC3, resize_image
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from share import *


def process(
    model,
    ddim_sampler,
    input_image,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    image_resolution,
    ddim_steps,
    guess_mode,
    strength,
    scale,
    seed,
    eta,
):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results


class ControlNetInferenceModel(pyd.BaseModel):
    prompt: t.Sequence[str] = pyd.Field(
        ..., description="Prompt to use for inference choosen randomly from this list"
    )
    added_prompt: str = pyd.Field(..., description="Added prompt to use for inference")
    negative_prompt: str = pyd.Field(
        ..., description="Negative prompt to use for inference"
    )
    num_samples: int = pyd.Field(1, description="Number of samples to generate")
    image_resolution: int = pyd.Field(
        512, description="Resolution of the generated images"
    )
    ddim_steps: int = pyd.Field(
        20, description="Number of DDIM steps to use for inference"
    )
    guess_mode: bool = pyd.Field(False, description="Whether to use guess mode or not")
    strength: float = pyd.Field(1.0, description="Strength of the control")
    scale: float = pyd.Field(9.0, description="Scale of the control")
    seed: int = pyd.Field(-1, description="Seed to use for inference")
    eta: float = pyd.Field(0.0, description="Eta to use for inference")


class ControlNet(pyd.BaseModel):
    config_path: str = pyd.Field(..., description="Path to the model config file")
    model_path: str = pyd.Field(..., description="Path to the model weights file")
    device: str = pyd.Field("cuda", description="Device to use for inference")

    _model: t.Any = pyd.PrivateAttr()
    _sampler: t.Any = pyd.PrivateAttr()

    def __init__(self, **data) -> None:
        super().__init__(**data)

        self._model = create_model(self.config_path).cpu()
        self._model.load_state_dict(
            load_state_dict(self.model_path, location=self.device)
        )
        self._model = self._model.to(self.device)
        self._sampler = DDIMSampler(self._model)

    def inference(
        self, image, inference_config: ControlNetInferenceModel
    ) -> t.List[np.ndarray]:
        prompt = random.choice(inference_config.prompt)
        output = process(
            self._model,
            self._sampler,
            image,
            prompt,
            inference_config.added_prompt,
            inference_config.negative_prompt,
            inference_config.num_samples,
            inference_config.image_resolution,
            inference_config.ddim_steps,
            inference_config.guess_mode,
            inference_config.strength,
            inference_config.scale,
            inference_config.seed,
            inference_config.eta,
        )
        # remove the first image which is the input image
        output = output[1:]
        # resize to original resolution
        output = [cv2.resize(img, (image.shape[1], image.shape[0])) for img in output]
        return output


class ControlNetStage(plst.SampleStage, title="control-net"):
    image_key: str = pyd.Field(..., description="Key of the image to use for inference")
    output_key: str = pyd.Field(..., description="Key to use for the output")

    config_path: str = pyd.Field(..., description="Path to the model config file")
    model_path: str = pyd.Field(..., description="Path to the model weights file")
    device: str = pyd.Field("cuda", description="Device to use for inference")
    config_inference: ControlNetInferenceModel = pyd.Field(
        ..., description="Inference config"
    )

    _controlnet: ControlNet = pyd.PrivateAttr()

    def __init__(self, **data) -> None:
        super().__init__(**data)

        self._controlnet = ControlNet(
            config_path=self.config_path,
            model_path=self.model_path,
            device=self.device,
        )

    def __call__(self, x: pls.Sample) -> pls.Sample:
        image = x[self.image_key]()
        output = self._controlnet.inference(image, self.config_inference)
        output = np.stack(output, axis=0)
        x = x.set_item(self.output_key, pli.TiffImageItem(output))
        return x
