ARG PQUEUE_VERSION
FROM pqueue:$PQUEUE_VERSION as pqueue

FROM appropriate/curl as model
RUN curl -o model.ckpt -L --insecure http://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt

FROM python:3.10 as libbuilder
WORKDIR /app
RUN pip install poetry git+https://github.com/python-poetry/poetry.git git+https://github.com/python-poetry/poetry-core.git
RUN python3.10 -m venv /app/venv 
# ENV PIP_FIND_LINKS=https://download.pytorch.org/whl/cu113/torch_stable.html
WORKDIR /app/
COPY ./pyproject.toml /app/
RUN VIRTUAL_ENV=/app/venv poetry install 

FROM ubuntu:jammy 
WORKDIR /app
RUN ln --symbolic --force --no-dereference /usr/share/zoneinfo/EST && echo "EST" > /etc/timezone
RUN apt update && DEBIAN_FRONTEND="noninteractive" apt install -y python3.10 git
RUN git clone https://github.com/CompVis/taming-transformers && mv taming-transformers/taming .
COPY --from=model /model.ckpt /app/models/ldm/text2img-large/model.ckpt 
COPY --from=libbuilder /app/venv/lib/python3.10/site-packages /app/
COPY ./configs /app/configs 
COPY ./data /app/data 
COPY ./ldm /app/ldm 
COPY ./models /app/models 
COPY --from=pqueue /src/pqueue /app/
COPY ./txt2img.py ./config.py ./run.py /app/
ENTRYPOINT ["/usr/bin/python3.10", "/app/run.py"]
