# Use the official TensorFlow GPU image from Docker Hub
#FROM tensorflow/tensorflow:2.14.0-gpu
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run your script when the container launches
CMD ["python","./training.py"]


