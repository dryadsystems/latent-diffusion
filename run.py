# Copyright (c) 2022 Dryad Systems
from pqueue import Maestro, Gen, Prompt, Result
import time
import logging
import txt2img


class DiffuseMaestro(Maestro):
    def create_generator(self) -> None:
        pass

    def handle_item(self, generator: Gen, prompt: Prompt) -> tuple[Gen, Result]:
        "finagle settings, generate it depending on settings, make a video if appropriate"
        args = txt2img.get_args({"prompt": prompt.prompt, **prompt.param_dict})
        logging.info(args)
        start_time = time.time()
        generator, path = txt2img.generate(generator, args)
        # return the generator so it can be reused
        return generator, Result(
            elapsed=round(time.time() - start_time),
            filepath=path,
            loss=-1,
            seed="",
        )


if __name__ == "__main__":
    DiffuseMaestro().main()
