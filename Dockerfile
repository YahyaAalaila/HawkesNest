# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

# Set the working directory
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install the package and its dependencies using uv
RUN uv pip install --system ".[all,dev]"

# Ensure the output directory exists
RUN mkdir -p outputs

# By default, provide a bash shell so the user can run scripts or the CLI.
# Alternatively, to run the CLI directly: ENTRYPOINT ["python", "-m", "hawkesnest.cli"]
CMD ["bash"]
