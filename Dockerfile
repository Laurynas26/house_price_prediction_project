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


# Download Sparticuz Chromium
RUN wget -O /tmp/chromium.tar \
    https://github.com/Sparticuz/chromium/releases/download/v141.0.0/chromium-v141.0.0-pack.x64.tar && \
    mkdir -p /opt/chromium && \
    tar -xf /tmp/chromium.tar -C /opt/chromium && \
    rm /tmp/chromium.tar

# Download matching chromedriver
RUN wget -O /tmp/chromedriver.zip \
    https://github.com/Sparticuz/chromium/releases/download/v141.0.0/chromedriver-v141.0.0-linux64.zip && \
    unzip /tmp/chromedriver.zip -d /opt && \
    mv /opt/chromedriver /usr/bin/chromedriver && \
    chmod +x /usr/bin/chromedriver && \
    rm /tmp/chromedriver.zip

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
