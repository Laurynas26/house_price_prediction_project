# Base AWS Lambda Python 3.10 image
FROM public.ecr.aws/lambda/python:3.10

# Install build tools (needed for xgboost and similar libs)
RUN yum install -y gcc g++ cmake make

# Copy Lambda function code
COPY src/aws_lambda/ ${LAMBDA_TASK_ROOT}/

# Copy src and config
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY config/ ${LAMBDA_TASK_ROOT}/config/

# Install dependencies
COPY src/aws_lambda/requirements_aws_lambda.txt .
RUN pip install --no-cache-dir -r requirements_aws_lambda.txt

# Set the Lambda handler (file.function)
CMD ["lambda_function.lambda_handler"]
