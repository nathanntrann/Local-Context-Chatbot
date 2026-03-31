FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY knowledge/ knowledge/

# Install the package and its dependencies
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "inspect_assist"]
