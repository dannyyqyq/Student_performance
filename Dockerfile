FROM python:3.10-slim-bullseye

WORKDIR /app

COPY . /app

# updates list of packages and intall AWS CLI
RUN apt update -y && apt install awscli -y

RUN pip install -r requirements.txt

CMD ["python", "application.py"]
