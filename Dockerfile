FROM python:3.8

# set the working dir in the container
WORKDIR /src

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ../flux/flux_modeller .
COPY unified_model .
COPY data .
COPY script.py .

CMD ["python", "./script.py"]
