# Fabricated Image Analyzer

A web app that checks whether a JPEG image file has been edited. You upload an image and it gives back a verdict, a confidence score, the photo's error level map, and metadata information (camera, software, GPS).

## How it works

Error level analysis (ELA) re-saves a JPEG at a fixed quality and subtracts the result from the original. A region that was pasted in or retouched usually has a different compression history from the rest of the picture, so it re-compresses differently and shows up brighter in the difference image.

The pipeline is:

1. Compute the ELA image, resize it to 224x224 and normalize it.
2. Run it through a convolutional neural network that outputs a single score.
3. Call the image tampered if the score is above a threshold chosen on the validation set.

## Results

I trained on the CASIA v2.0 dataset ([Dataset](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset)) and kept only the JPEG files: 9,501 images, 7,437 authentic and 2,064 tampered. Tampered images in CASIA are made from authentic ones, so a random split would leak. I grouped images by the source photo in the filename and split by group, 70/15/15 (6,661 train, 1,431 validation, 1,409 test). It gave an accuracy of 96%

## Running it

### Train

    python train.py

This builds `data/manifest.csv` if it is missing, trains the model, writes the run (weights, `metrics.json`, `history.csv`) to a folder under `runs/` and saves a copy for the backend in `models/`.

### Run the app

    python -m uvicorn backend.main:app --reload

In a second terminal:

    cd frontend
    npm install
    npm run dev

Open http://localhost:5173

## Layout

    forgery/    ELA, data prep, models, metrics, export
    backend/    FastAPI service that loads the trained model
    frontend/   React + Vite app
    train.py    prepares data, trains, saves the model

The app doesn't store uploaded images.

## Screenshots

<img width="1919" height="945" alt="image" src="https://github.com/user-attachments/assets/72fa8139-a28f-4a6a-95af-f389ab6068a5" />

<img width="1919" height="945" alt="image" src="https://github.com/user-attachments/assets/890af156-4f77-417a-a8db-ad952f642df5" />
