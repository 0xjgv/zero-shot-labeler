FROM public.ecr.aws/lambda/python:3.11

# Install poetry
RUN pip install poetry==1.8.4 && \
  poetry config virtualenvs.create false

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-cache --no-root --only main

# Copy application code
COPY zero_shot_labeler ${LAMBDA_TASK_ROOT}/zero_shot_labeler/

# Make the model directory and run the preload script in builder
RUN mkdir -p /opt/ml/model && \
  poetry run python zero_shot_labeler/__init__.py

# Set Python path
ENV PYTHONPATH "${LAMBDA_TASK_ROOT}"

# Remove the cache directory
RUN rm -rf /root/.cache/

# Lambda handler
CMD [ "zero_shot_labeler.lambda_handler.handler" ]