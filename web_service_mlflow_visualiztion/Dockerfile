# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Update package index and install necessary packages
RUN apt-get update \
    && apt-get install -y libgomp1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file from the host machine into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model_ag_deployment directory from the host machine into the container at /app/model_ag_deployment
# COPY model_ag_deployment /app/model_ag_deployment

# Copy the modules directory from the host machine into the container at /app/modules
COPY modules /app/modules

# Copy the predict.py file from the host machine into the container at /app
COPY predict.py /app/
COPY .env /app/

# Copy the download_folder_s3.py file from the host machine into the container at /app
COPY download_folder_s3.py /app/

# Expose the port on which the app will run
EXPOSE 9696

# Command to run the application using Gunicorn
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]
# CMD ["bash"]
