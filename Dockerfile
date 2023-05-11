FROM python:3.7.9

WORKDIR /src

COPY requirement.txt  .

RUN pip install -r requirement.txt

COPY .  .

CMD ["python","./src/app.py"]
