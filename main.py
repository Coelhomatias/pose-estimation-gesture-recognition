import cv2
from ultralytics import YOLO
import numpy as np
import time
import serial

# Configure the serial port
ser = serial.Serial('COM4', 2000000, timeout=1)

# Initialize global variables
last_zone_entry_time = 0
in_zone = False
robot_moving_left = False
robot_moving_right = False
pickup_initiated = False
left_arm_raised = 0
right_arm_raised = 0
last_message_time = 0
rate_limit_interval = 0.01  # Rate limit interval in seconds

def close_serial():
    ser.close()
    pass

def rate_limited(interval):
    """Rate-limited decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            global last_message_time
            current_time = time.time()
            if current_time - last_message_time >= interval:
                last_message_time = current_time
                return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limited(rate_limit_interval)
def move_arm_left():
    message = '1'
    print("Moving arm left")
    ser.write(message.encode())

@rate_limited(rate_limit_interval)
def move_arm_right():
    message = '2'
    print("Moving arm right")
    ser.write(message.encode())

@rate_limited(rate_limit_interval)
def pickup_cube():
    message = '3'
    print("Picking up cube")
    ser.write(message.encode())

def draw_interaction_zone(frame, x_start, x_end, color=(255, 0, 0), thickness=2):
    """Draw vertical lines on the image to define an interaction zone."""
    cv2.line(frame, (x_start, 0), (x_start, frame.shape[0]), color, thickness)
    cv2.line(frame, (x_end, 0), (x_end, frame.shape[0]), color, thickness)
    return frame

def is_within_zone(x, x_start, x_end):
    """Check if a point is within the defined vertical zone."""
    return x_start <= x <= x_end

def calculate_angle(pointA, pointB, pointC):
    """
    Calculate the angle between three points (A, B, C) where B is the vertex.
    
    Parameters:
    pointA (tuple): Coordinates of the first point (x1, y1).
    pointB (tuple): Coordinates of the second point (x2, y2).
    pointC (tuple): Coordinates of the third point (x3, y3).
    
    Returns:
    float: Angle in degrees between the lines AB and BC.
    """
    # Convert points to numpy arrays
    A = np.array(pointA)
    B = np.array(pointB)
    C = np.array(pointC)
    
    # Create vectors AB and BC
    AB = A - B
    BC = C - B
    
    # Calculate the dot product and magnitudes of AB and BC
    dot_product = np.dot(AB, BC)
    magnitude_AB = np.linalg.norm(AB)
    magnitude_BC = np.linalg.norm(BC)
    
    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_AB * magnitude_BC)
    
    # Calculate the angle in radians and then convert to degrees
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def gesture_control(person, x_start, x_end, frame):
    global last_zone_entry_time, in_zone, robot_moving_left, robot_moving_right, pickup_initiated, left_arm_raised, right_arm_raised
    
    """Check person's keypoints to determine gestures and control robot."""
    person = person.xy.squeeze().cpu()

    # Example keypoints indices for shoulders and elbows
    left_shoulder, right_shoulder = person[5], person[6]
    left_elbow, right_elbow = person[7], person[8]
    left_wrist, right_wrist = person[9], person[10]
    left_waist, right_waist = person[11], person[12]

    current_time = time.time()

    frame = draw_interaction_zone(frame, x_start, x_end, color=(0, 0, 255))

    # Check if person is within the interaction zone
    if is_within_zone(left_shoulder[0], x_start, x_end) and is_within_zone(right_shoulder[0], x_start, x_end):
        if not in_zone:
            last_zone_entry_time = current_time
            in_zone = True
        elif current_time - last_zone_entry_time >= 1:  # 1 second delay
            frame = draw_interaction_zone(frame, x_start, x_end, color=(0, 255, 0))

            # Calculate angles
            left_arm_angle = calculate_angle(left_waist, left_shoulder, left_elbow)
            right_arm_angle = calculate_angle(right_waist, right_shoulder, right_elbow)

            left_forearm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_forearm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Define thresholds for gestures
            if left_arm_angle > 60:
                if not left_arm_raised:
                    left_arm_raised = current_time

                if not robot_moving_right and current_time - left_arm_raised >= 0.5 and not pickup_initiated:  # Example threshold for "arm raised" with delay
                    move_arm_left()
                    robot_moving_left = True
            else:
                left_arm_raised = 0
                robot_moving_left = False

            if right_arm_angle > 60:
                if not right_arm_raised:
                    right_arm_raised = current_time

                if not robot_moving_left and current_time - right_arm_raised >= 0.5 and not pickup_initiated:  # Example threshold for "arm raised" with delay
                    move_arm_right()
                    robot_moving_right = True
            else:
                right_arm_raised = 0
                robot_moving_right = False
            
            if ((left_arm_angle > 110 and left_forearm_angle < 90) or (right_arm_angle > 110 and right_forearm_angle < 90)):
                if not pickup_initiated:
                    pickup_cube()
                    pickup_initiated = True
            elif left_arm_angle <= 60 and right_arm_angle <= 60:
                pickup_initiated = False
    else:
        in_zone = False


    return frame

def run_tracker(filename, model):
    video = cv2.VideoCapture(filename)

    # Create a window named "Tracking_Stream" and set it to fullscreen
    cv2.namedWindow("Tracking_Stream", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Tracking_Stream", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = video.read()
        if not ret:
            break


        results = model.track(frame, persist=True, verbose=False)
        res_plotted = results[0].plot()
        
        if len(results[0].keypoints) == 1 and results[0].keypoints[0].has_visible:
            gesture_control(results[0].keypoints[0], 150, 490, res_plotted)

        cv2.imshow("Tracking_Stream", res_plotted)
        key = cv2.waitKey(1)
        if key == ord('q'):
            close_serial()
            break

    video.release()

# Load the models
model = YOLO("models/yolov8m-pose.engine")
video_file = 0  # For webcam input
run_tracker(video_file, model)
cv2.destroyAllWindows()
