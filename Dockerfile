# Base AWS Lambda Python 3.10 image
FROM public.ecr.aws/lambda/python:3.10

# -----------------------------
# Install system dependencies
# -----------------------------
RUN yum -y update && \
    yum install -y \
        unzip \
        wget \
        nss \
        freetype \
        fontconfig \
        alsa-lib \
        atk \
        cups-libs \
        gtk3 \
        GConf2 \
        xdg-utils \
        libX11 \
        libXcomposite \
        libXcursor \
        libXdamage \
        libXext \
        libXi \
        libXrandr \
        libXScrnSaver \
        libXtst \
        pango \
    && yum clean all

# -----------------------------
# Install Chromium (version 128)
# -----------------------------
RUN wget -O chrome.zip \
    https://storage.googleapis.com/chrome-for-testing-public/128.0.6613.137/linux64/chrome-linux64.zip && \
    unzip chrome.zip && \
    mv chrome-linux64 /opt/chrome && \
    rm chrome.zip

# -----------------------------
# Install matching Chromedriver
# -----------------------------
RUN wget -O chromedriver.zip \
    https://storage.googleapis.com/chrome-for-testing-public/128.0.6613.137/linux64/chromedriver-linux64.zip && \
    unzip chromedriver.zip && \
    mv chromedriver-linux64/chromedriver /usr/bin/chromedriver && \
    chmod +x /usr/bin/chromedriver && \
    rm -rf chromedriver-linux64 chromedriver.zip

# -----------------------------
# Set environment variables for Selenium
# -----------------------------
ENV CHROME_PATH="/opt/chrome/chrome"
ENV CHROMEDRIVER_PATH="/usr/bin/chromedriver"
ENV PATH="$PATH:/opt/chrome:/usr/bin"

# -----------------------------
# Working directory
# -----------------------------
WORKDIR ${LAMBDA_TASK_ROOT}

# -----------------------------
# Copy code and config
# -----------------------------
COPY src/aws_lambda/ ${LAMBDA_TASK_ROOT}/
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY config/ ${LAMBDA_TASK_ROOT}/config/

# -----------------------------
# Install Python dependencies
# -----------------------------
COPY src/aws_lambda/requirements_aws_lambda.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "numpy<2" -r requirements_aws_lambda.txt

# -----------------------------
# Lambda handler
# -----------------------------
CMD ["lambda_function.lambda_handler"]
