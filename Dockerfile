FROM public.ecr.aws/lambda/python:3.11

# Install poetry
RUN pip install poetry==1.8.4 && \
    poetry config virtualenvs.create false

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --only main

# Copy application code
COPY playground ${LAMBDA_TASK_ROOT}/playground/

# Run preload script in builder
RUN mkdir -p /opt/ml/model && \
    poetry run preload

# Set Python path
ENV PYTHONPATH "${LAMBDA_TASK_ROOT}"

# Lambda handler
CMD [ "playground.lambda_handler.handler" ]