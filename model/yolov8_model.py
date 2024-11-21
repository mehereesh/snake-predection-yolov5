from ultralytics import YOLO  # Import YOLO from Ultralytics

def detect_snake(image_path):
    # Load the pre-trained YOLOv8 model
    model = YOLO("C:/snake_detection/model/weights/yolov8n.pt")  # Replace with the correct weight file path

    # Run inference on the given image or video
    results = model(image_path)

    # Show the results with bounding boxes around detected objects
    results.show()

    # Optionally, save the results to a file
    results.save("path_to_save_results")  # Replace with the path where you want to save the result

if __name__ == "__main__":
    # Example usage: provide an image or video path
    detect_snake("path_to_your_image_or_video")  # Modify with the correct input path
