FROM python:3.13-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY knowledge/ knowledge/

# Create data directory for SQLite DB and reports
RUN mkdir -p data/images data/reports

# Install the package and its dependencies
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "inspect_assist"]
