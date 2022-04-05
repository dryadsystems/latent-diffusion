FROM appropriate/curl as model
RUN curl -o model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt

FROM python:3.9 as libbuilder
WORKDIR /app
RUN pip install poetry
RUN python3.9 -m venv /app/venv 
# ENV PIP_FIND_LINKS=https://download.pytorch.org/whl/cu113/torch_stable.html
COPY ./pyproject.toml ./poetry.lock /app/
RUN VIRTUAL_ENV=/app/venv poetry install 

FROM ubuntu:hirsute 
WORKDIR /app
RUN ln --symbolic --force --no-dereference /usr/share/zoneinfo/EST && echo "EST" > /etc/timezone
RUN apt update && DEBIAN_FRONTEND="noninteractive" apt install -y python3.9
COPY --from=libbuilder /app/venv/lib/python3.9/site-packages /app/
COPY --from=model /model.ckpt /app/models/ldm/text2img-large/model.ckpt 
COPY ./configs ./data ./ldm ./main.py ./models ./scripts /app/
ENTRYPOINT ["/app/scripts/txt2img.py"]
