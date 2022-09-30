# Copyright (c) 2022 Dryad Systems
import logging
import time
from typing import Any, Iterator
import requests
import torch
from omegaconf import OmegaConf
from PIL.Image import Image
import txt2img
from go_brrr import BrrrGoer

logging.getLogger().setLevel("DEBUG")


class DiffuseGoBrrr(BrrrGoer[Any]):
    model = "diffuse"

    def __init__(self) -> None:
        self.session = requests.Session()
        super().__init__()

    def create_generator(self) -> Any:
        config = OmegaConf.load(
            "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        )  # TODO: Optionally download from same location as ckpt and chnage this logic
        model = txt2img.load_model_from_config(  # type: ignore
            config, "models/ldm/text2img-large/model.ckpt"
        )  # TODO: check path

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logging.info(device)
        model = model.to(device)
        return model

    def handle_item(self, generator: Any, params: dict) -> Iterator[Image | dict]:
        args = txt2img.get_args({"prompt": params["prompts"][0]["text"], **params})
        logging.info(args)
        params["H"] = params.get("height", 256)
        params["W"] = params.get("width", 256)
        params["n_samples"] = params.get("num_images", 1)
        start_time = time.time()
        images = txt2img.generate(generator, args)
        yield from images
        yield {
            "elapsed": round(time.time() - start_time, 4),
        }


if __name__ == "__main__":
    DiffuseGoBrrr().run()
