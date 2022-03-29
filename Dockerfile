FROM python:3.9-slim-bullseye as base

ENV PYTHONUNBUFFERED=1

# Install wget to set up the PPA (Personal Package Archives),
# xvfb to have a virtual screen, and unzip to install the
# Chromedriver, among other packages
RUN apt-get update -y && apt-get install -y wget xvfb unzip make gnupg curl apt-transport-https
RUN apt-get update

# Set up the Chrome PPA
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list

WORKDIR /app

# Update the package list and install chrome
RUN apt-get update -y
RUN apt-get install -y google-chrome-stable

# Set up Chromedriver environment variables
ENV CHROMEDRIVER_VERSION 95.0.4638.54
ENV CHROMEDRIVER_DIR /usr/bin

# Install Chromedriver
RUN mkdir -p $CHROMEDRIVER_DIR
RUN wget -q --continue -P $CHROMEDRIVER_DIR "http://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip"
RUN unzip $CHROMEDRIVER_DIR/chromedriver* -d $CHROMEDRIVER_DIR


# install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm


# ENTRYPOINT [ "python", "./app/run.py"]

# # copy app files
FROM base as prod
COPY exsclaim /app/
