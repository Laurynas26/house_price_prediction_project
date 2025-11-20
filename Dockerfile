FROM public.ecr.aws/lambda/python:3.10

# Install dependencies
RUN yum -y update && \
    yum install -y \
        tar \
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


# Download Sparticuz Chromium + Chromedriver
RUN wget -O /tmp/chromium.zip \
    https://github.com/Sparticuz/chromium/releases/download/v141.0.0/chromium-v141.0.0-layer.x64.zip && \
    mkdir -p /opt/chromium && \
    unzip /tmp/chromium.zip -d /opt/chromium && \
    rm /tmp/chromium.zip

# Make chromedriver executable
RUN chmod +x /opt/chromium/chromedriver && \
    mv /opt/chromium/chromedriver /usr/bin/chromedriver

# Set environment variables
ENV CHROME_PATH="/opt/chromium/chrome"
ENV CHROMEDRIVER_PATH="/usr/bin/chromedriver"
ENV PATH="$PATH:/opt/chromium:/usr/bin"

# Working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy code
COPY src/aws_lambda/ ${LAMBDA_TASK_ROOT}/
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY config/ ${LAMBDA_TASK_ROOT}/config/

# Python dependencies
COPY src/aws_lambda/requirements_aws_lambda.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements_aws_lambda.txt

CMD ["lambda_function.lambda_handler"]
