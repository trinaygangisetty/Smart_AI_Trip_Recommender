# Use the official Python image from Docker Hub
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy your project files to the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Flask runs on
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]
