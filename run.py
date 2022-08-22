# Copyright (c) 2022 Dryad Systems
from pqueue import Maestro, Gen, Prompt, Result
import time
import logging
import txt2img


class DiffuseMaestro(Maestro):
    model = "diffuse"
    version = "0.0.24"

    def create_generator(self) -> None:
        pass

    def handle_item(self, generator: Gen, prompt: Prompt) -> tuple[Gen, Result]:
        "finagle settings, generate it depending on settings, make a video if appropriate"
        args = txt2img.get_args(
            {"prompt": prompt.params["prompts"][0]["text"], **prompt.params}
        )
        logging.info(args)
        prompt.params["H"] = prompt.params.get("height", 256)
        prompt.params["W"] = prompt.params.get("width", 256)
        prompt.params["n_samples"] = prompt.params.get("num_images", 1)
        start_time = time.time()
        generator, images = txt2img.generate(generator, args)
        # return the generator so it can be reused
        return generator, Result(
            elapsed=round(time.time() - start_time),
            images=images,
            loss=-1,
            seed="",
        )


if __name__ == "__main__":
    DiffuseMaestro().main()
