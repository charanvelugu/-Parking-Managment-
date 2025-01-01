import cv2
from parking import ParkingManagement

# Video capture
input_video_path = r"C:\Users\HII\Desktop\Charan_projects\yolo11-parkinglot-main\parking1.mp4"
output_video_path = r"C:\Users\HII\Desktop\Charan_projects\output_parking_hd.mp4"
cap = cv2.VideoCapture(input_video_path)

# Get the frame width, height, and frames per second (fps) from the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Use HD resolution for the output video
output_width = 1920
output_height = 1080

# Initialize video writer for saving the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (1080, 600))

# Initialize parking management object
parking_manager = ParkingManagement(
    model="yolo11s.pt",  # path to model file
    classes=[2],
    json_file="bounding_boxes.json",  # path to parking annotations file
)

while cap.isOpened():
    ret, im0 = cap.read()
    if not ret:
        break

    # Resize the frame for processing
    im01 = cv2.resize(im0, (1080, 600))

    # Process the frame using parking manager
    im0_processed = parking_manager.process_data(im01)

    # Write the processed frame to the output video
    out.write(im0_processed)

    # Display the processed frame
    cv2.imshow("im0", im0_processed)

    # Break the loop on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
