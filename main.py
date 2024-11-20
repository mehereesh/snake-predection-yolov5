import tkinter as tk
from tkinter import filedialog
import cv2
import math
import threading
from twilio.rest import Client
from ultralytics import YOLO
import cvzone

# Twilio account SID and Auth Token (replace with your credentials)
account_sid = 'ACa18593aa83c74e7230b5e1ab7f9b1dc9'  # Your Twilio Account SID
auth_token = 'fbe48629c453bb1c0501ac1c9fbbf278'    # Your Twilio Auth Token
twilio_phone_number = '+14302224933'  # Example: '+14155552671' (replace with your Twilio number)
your_phone_number = '+918142811718'    # Your phone number (replace with your actual number)

# Initialize Twilio client
client = Client(account_sid, auth_token)

cap = None
processing = False

def send_sms_alert():
    """Function to send SMS alert using Twilio"""
    try:
        message = client.messages.create(
            body="Alert: Snake detected in the webcam feed!",
            from_=twilio_phone_number,  # Twilio phone number
            to=your_phone_number        # Your phone number
        )
        print(f"Message sent successfully: {message.sid}")
    except Exception as e:
        print(f"Error sending SMS: {e}")

def detect_from_file():
    """Function to browse and load a video file for detection (No SMS)"""
    global cap
    filepath = filedialog.askopenfilename()
    if filepath:
        if cap is not None:
            cap.release()  # Release any previously opened video capture
        cap = cv2.VideoCapture(filepath)
        threading.Thread(target=run_detection, args=(False,)).start()  # Start video detection without SMS

def detect_from_webcam():
    """Function to use the webcam for real-time snake detection (With SMS)"""
    global cap
    if cap is not None:
        cap.release()  # Release any previously opened video capture
    cap = cv2.VideoCapture(0)
    threading.Thread(target=run_detection, args=(True,)).start()  # Start webcam detection with SMS

def stop_processing():
    """Function to stop the detection process"""
    global processing
    processing = False  # Set processing flag to False to stop the detection loop

def run_detection(use_sms):
    """Function to run snake detection using YOLO"""
    global cap, processing
    model = YOLO(r'C:\Users\meher.DELL\Downloads\snake_detetion\snake_detetion\best.pt')  # Load your YOLO model
    classnames = ['Snake']  # Assuming class 'Snake' is defined in your model
    processing = True
    
    while processing:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))  # Resize frame to fit the model
            result = model(frame, stream=True)  # Run YOLO on the frame
            snake_detected = False

            for info in result:
                boxes = info.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    confidence = math.ceil(confidence * 100)
                    Class = int(box.cls[0])
                    if confidence > 50:  # Only detect with confidence above 50%
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Draw bounding box
                        cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                           scale=1.5, thickness=2)  # Add text with confidence percentage
                        snake_detected = True  # Flag to indicate snake detection

            cv2.imshow('Snake Detection', frame)  # Show the frame with detection

            # Send SMS alert only for real-time webcam detection
            if snake_detected and use_sms:
                threading.Thread(target=send_sms_alert).start()  # Send SMS alert if snake detected

            if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to stop processing
                break

    cv2.destroyAllWindows()  # Close all OpenCV windows
    if cap is not None:
        cap.release()  # Release video capture

# Tkinter GUI
root = tk.Tk()
root.title("Snake Detection System")
root.geometry("400x250")

# Buttons to control detection
browse_button = tk.Button(root, text="Browse Video File", command=detect_from_file)
browse_button.pack(pady=10)

webcam_button = tk.Button(root, text="Use Webcam", command=detect_from_webcam)
webcam_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Detection", command=stop_processing)
stop_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
