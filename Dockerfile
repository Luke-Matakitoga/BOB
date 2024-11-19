# Use an ROCm-compatible base image
FROM rocm/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy the chatbot script and requirements file into the container
COPY main.py /app/
COPY requirements.txt /app/

# Install additional Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port (optional for future extensions)
EXPOSE 8080

# Entry point to run the chatbot
CMD ["python", "main.py"]
