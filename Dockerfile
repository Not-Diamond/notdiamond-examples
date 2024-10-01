# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install dependencies (curl, build-essential, libpq-dev)
RUN apt-get update && apt-get install -y curl build-essential libpq-dev

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set poetry path
ENV PATH="/root/.local/bin:$PATH"

# Copy the current directory contents into the container
COPY . /app

# Set the working directory
WORKDIR /app

# Install Python dependencies using Poetry
RUN /root/.local/bin/poetry install

# Copy .env file to container
COPY .env /app

# Expose the port on which the app will run
EXPOSE 8501

# Run the streamlit app using Poetry's virtualenv path
CMD ["/root/.local/bin/poetry", "run", "streamlit", "run", "notdiamond_examples/streamlit/main.py"]
