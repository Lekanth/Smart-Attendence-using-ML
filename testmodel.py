import cv2
import csv
import os
import time
from datetime import datetime
from screeninfo import get_monitors

# Get the display resolution
monitor = get_monitors()[0]
display_width = monitor.width
display_height = monitor.height

# Initialize the video capture for webcam
video = cv2.VideoCapture(0)

# Initialize the video capture for the background video (play at 4x speed)
background_video = cv2.VideoCapture("elder.mp4")

# Check if the background video is opened successfully
if not background_video.isOpened():
    print("Error: Could not open background video.")
    exit()

# Load the face detection and recognition models
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

# List of names corresponding to the labels in the recognizer
name_list = [" ","nisarga","naga"," sir"]
class_list = [ " ","Class B","Class C","class A"]
branch_list = [" ","Branch Y","branch z","branch s"]

# Column names for the CSV file
COL_NAMES = ['NAME', 'CLASS', 'BRANCH', 'TIME']

# Create the Attendance directory if it doesn't exist
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

while True:
    ret_bg, bg_frame = background_video.read()
    if not ret_bg:
        background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_bg, bg_frame = background_video.read()

    # Check if the background frame is empty
    if bg_frame is None or bg_frame.size == 0:
        print("Warning: Background frame is empty or not read properly.")
        continue

    ret, frame = video.read()
    if not ret or frame is None or frame.size == 0:
        print("Warning: Unable to read frame from webcam.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # Ensure the sliced image is valid
        if gray[y:y + h, x:x + w].size > 0:
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            csv_path = "Attendance/Attendance_" + date + ".csv"
            exist = os.path.isfile(csv_path)
            
            if conf > 50 and serial < len(name_list):
                name = name_list[serial]
                class_name = class_list[serial]
                branch_name = branch_list[serial]
            else:
                name = "Unknown"
                class_name = "Unknown"
                branch_name = "Unknown"
            
            text = f"{name}, {class_name}, {branch_name}"
            # Create a background for the text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x + 5
            text_y = y - 5
            if text_y - text_size[1] - 5 < 0:
                text_y = y + h + text_size[1] + 5
            # Ensure text background does not exceed rectangle frame
            if text_x + text_size[0] + 5 > x + w:
                text_x = x + w - text_size[0] - 5
            if text_y - text_size[1] - 5 < y:
                text_y = y + text_size[1] + 5
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 255), -1)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            attendance = [name, class_name, branch_name, timestamp]

    frame = cv2.resize(frame, (480, 360))  # Make the frame smaller
    
    # Resize the background frame to fit the display resolution
    if bg_frame is not None and bg_frame.size > 0:
        bg_frame = cv2.resize(bg_frame, (display_width, display_height))
        
        bg_height, bg_width, _ = bg_frame.shape
        if bg_height < 162 + 360 or bg_width < 55 + 480:
            new_bg_height = max(bg_height, 162 + 360)
            new_bg_width = max(bg_width, 55 + 480)
            bg_frame = cv2.resize(bg_frame, (new_bg_width, new_bg_height))
        
        bg_frame[162:162 + 360, 55:55 + 480] = frame  # Update position to fit the smaller frame
        cv2.imshow("Frame", bg_frame)
    else:
        print("Warning: Background frame is empty after second check.")
    
    k = cv2.waitKey(1)
    
    if k == ord('o'):
        with open(csv_path, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)

    if k == ord("q"):
        break

video.release()
background_video.release()
cv2.destroyAllWindows()
