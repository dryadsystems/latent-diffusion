ARG PQUEUE_VERSION
FROM pqueue:$PQUEUE_VERSION as pqueue

FROM appropriate/curl as model
RUN curl -o model.ckpt -L https://r2-public-worker.drysys.workers.dev/text2img-large.ckpt

FROM python:3.10 as libbuilder
WORKDIR /app
RUN pip install poetry git+https://github.com/python-poetry/poetry.git git+https://github.com/python-poetry/poetry-core.git
RUN python3.10 -m venv /app/venv 
# ENV PIP_FIND_LINKS=https://download.pytorch.org/whl/cu113/torch_stable.html
WORKDIR /app/
COPY ./pyproject.toml /app/
RUN VIRTUAL_ENV=/app/venv poetry install 

FROM timberio/vector:nightly-2022-09-01-debian as vector

FROM python:3.10
WORKDIR /app
RUN git clone https://github.com/CompVis/taming-transformers && mv taming-transformers/taming .
COPY --from=model /model.ckpt /app/models/ldm/text2img-large/model.ckpt 
COPY --from=libbuilder /app/venv/lib/python3.10/site-packages /app/
COPY --from=vector /usr/bin/vector /app/vector
COPY ./configs /app/configs 
COPY ./data /app/data 
COPY ./ldm /app/ldm 
COPY ./models /app/models 
COPY --from=pqueue /src/pqueue /app/
COPY ./txt2img.py ./brrr.py /app/
ARG MODEL_VERSION
ENV MODEL_VERSION=$MODEL_VERSION
ENV MODEL="diffuse"
ENTRYPOINT ["/usr/local/bin/python3.10", "/app/pqueue.py"]
