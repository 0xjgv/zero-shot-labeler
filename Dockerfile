FROM public.ecr.aws/lambda/python:3.11 as builder

# Install poetry in builder stage
RUN pip install poetry==1.8.4 && \
    poetry config virtualenvs.create false

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies into site-packages
RUN poetry install --only main

# Copy application code
COPY playground ${LAMBDA_TASK_ROOT}/playground/

# Run preload script in builder
RUN mkdir -p /opt/ml/model && \
    poetry run preload

# Final stage
FROM public.ecr.aws/lambda/python:3.11

# Copy only the installed packages from builder
COPY --from=builder /var/lang/lib/python3.11/site-packages/ /var/lang/lib/python3.11/site-packages/

# Copy application code and model
COPY --from=builder ${LAMBDA_TASK_ROOT}/playground ${LAMBDA_TASK_ROOT}/playground/
COPY --from=builder /opt/ml/model /opt/ml/model/

# Set Python path
ENV PYTHONPATH "${LAMBDA_TASK_ROOT}"

# Lambda handler
CMD [ "playground.lambda_handler.handler" ]