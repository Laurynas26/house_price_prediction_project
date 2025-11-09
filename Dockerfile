# Base AWS Lambda Python 3.10 image
FROM public.ecr.aws/lambda/python:3.10

# Install dependencies for Chromium and Selenium
RUN yum -y update && \
    yum install -y \
        chromium \
        nss \
        freetype \
        fontconfig \
        alsa-lib \
        atk \
        gtk3 \
        ipa-gothic-fonts \
        wget \
        unzip \
        xorg-x11-fonts-100dpi \
        xorg-x11-fonts-75dpi \
        xorg-x11-utils \
        xorg-x11-fonts-cyrillic \
        xorg-x11-fonts-Type1 \
        xorg-x11-fonts-misc \
        libX11 \
        libXcomposite \
        libXcursor \
        libXdamage \
        libXext \
        libXi \
        libXrandr \
        libXScrnSaver \
        libXtst \
        libXxf86vm \
        cups-libs \
        pango \
        GConf2 \
        atk \
        gtk2 \
        gtk3 \
        libgdk-pixbuf2.0-0 \
        libxkbfile \
        mesa-libOSMesa \
        xdg-utils \
    && yum clean all

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy Lambda function code and config
COPY src/aws_lambda/ ${LAMBDA_TASK_ROOT}/
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY config/ ${LAMBDA_TASK_ROOT}/config/

# Copy and install Python dependencies
COPY src/aws_lambda/requirements_aws_lambda.txt .
RUN pip install --no-cache-dir -r requirements_aws_lambda.txt

# Set environment variables for headless Chromium
ENV PATH="/usr/bin/chromium:${PATH}"
ENV CHROME_BIN="/usr/bin/chromium"

# Lambda handler
CMD ["lambda_function.lambda_handler"]
