FROM python:3-alpine3.15

WORKDIR /app

# Install system dependencies
RUN apk add --no-cache build-base libffi-dev

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Command to run the application
CMD ["python", "./model_training.py"]
