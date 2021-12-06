FROM python:3.9

# set the working dir in the container
WORKDIR /src

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY data/2019-05-23_D/A data/2019-05-23_D/A
COPY data/flux_curve_model data/flux_curve_model
COPY data/magnetic-spring data/magnetic-spring

COPY flux_modeller flux_modeller
RUN cd flux_modeller && pip install .

COPY unified_model unified_model
COPY setup.py .
RUN pip install .

COPY *.py ./
RUN true  # prevents copying from failing
COPY scripts scripts

# -u flag flushes buffers so print statements showup
ENTRYPOINT ["python", "-u"]
