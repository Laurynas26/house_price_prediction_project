# Base AWS Lambda Python 3.10 image
FROM public.ecr.aws/lambda/python:3.10

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy your function code and supporting files
COPY src/aws_lambda/ ${LAMBDA_TASK_ROOT}/
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY config/ ${LAMBDA_TASK_ROOT}/config/

# Install Python dependencies directly (no CMake, no build tools)
# Use prebuilt wheel for xgboost (2.0.3) to avoid compilation
RUN pip install --no-cache-dir pandas numpy pyyaml xgboost==2.0.3

# Set the Lambda handler (file.function)
CMD ["lambda_function.lambda_handler"]
