# Overview

Images for Lenovo, Macbook and VR devices are included, however due to the size of the trained model, the model has not been included. 

After cloning, you can train a new model by running this command from the root.

python3 scripts/retrain.py --image_dir=tf_files/device_photos

One the model has been trained. From root, you can run the following to start the Python server.

python3 scripts/server.py

Point requests to the port of this server locally to get it to recognise an image and return the image type to you.
