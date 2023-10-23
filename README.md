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

2. Activate the virtual environment:
   
   ```bash
   # On macOS and Linux
   source venv/bin/activate
   
   # On Windows
   venv\Scripts\activate

3. Install project dependencies:
   
   ```bash
   pip install -r requirements.txt

4. Configure the database connection in my_db_other.py. Update the host, user, database, and passwd settings to match your MySQL configuration.

5. Create a MySQL database and table. Run the following SQL commands to create the anpr table:

   ```sql
   CREATE DATABASE parking; -- Create the 'parking' database
   USE parking; -- Use the 'parking' database
   
   CREATE TABLE anpr (
       EMP_ID INT PRIMARY KEY,
       NAME VARCHAR(250) NOT NULL,
       PHONE_NO INT NOT NULL,
       VEHICLE_NO VARCHAR(250)
   );

6. Registration data, including employee IDs, names, phone numbers, and vehicle numbers, are stored in the *anpr* table in the *parking* database.
   
7. Run the application:

   ```bash
   python -m app.py

