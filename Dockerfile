# small example dockerfile
FROM python:3.6
COPY ./src/ /code
COPY requirements.txt /temp/
WORKDIR /code
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /temp/requirements.txt

# run a training script or similar
CMD ["python", "models/train_model.py"]
