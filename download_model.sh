#!/bin/bash

if [ ! -f shape_predictor_68_face_landmarks.dat ]; then
    echo "Downloading face landmark model..."
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    echo "Extracting model file..."
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
    echo "Model file ready!"
else
    echo "Model file already exists. Skipping download."
fi