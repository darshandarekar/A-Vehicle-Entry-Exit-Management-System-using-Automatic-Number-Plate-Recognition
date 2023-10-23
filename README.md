# A-Vehicle-Entry-Exit-Management-System-using-Automatic-Number-Plate-Recognition

## Introduction

This project is an Optical Character Recognition (OCR) and Automatic Number Plate Recognition (ANPR) system. It detects license plates in images and videos, extracts the text from the plates, and stores the data in a MySQL database.

## Requirements

To run this project, you need the following:

- Python 3.x
- OpenCV (Open Source Computer Vision Library)
- PaddleOCR for text extraction
- MySQL (to store registration data)
- ONNX Model for YOLO (You Only Look Once) object detection

## Setup

Follow these steps to set up and run the project:

1. Create a virtual environment (optional but recommended):
   
   ```bash
   # On macOS and Linux
   python3 -m venv venv

   # On Windows
   python -m venv venv
