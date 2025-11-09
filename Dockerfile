# Base AWS Lambda Python 3.10 image
FROM public.ecr.aws/lambda/python:3.10

# Copy Lambda function code
COPY aws_lambda/ ${LAMBDA_TASK_ROOT}/

# Copy src and config
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY config/ ${LAMBDA_TASK_ROOT}/config/

# Install dependencies
COPY aws_lambda/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set the Lambda handler (file.function)
CMD ["lambda_function.lambda_handler"]
