# Use the Python3.7.2 container image
FROM python:3.7.2-stretch

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install the dependencies
# These commands are run inside the docker container.
# Docker builds a seperate image each time it sees a RUN command.
# Therefore use the && command after the first RUN.
# Video 30

RUN pip3 install torch torchvision --no-cache-dir

# The next 4 are required to fix errors with cv2
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN pip install --upgrade pip
RUN pip install opencv-python
RUN apt-get update
RUN apt install -y libgl1-mesa-glx

RUN pip install -r requirements.txt

# run the command to start uWSGI
CMD ["uwsgi", "app.ini"]