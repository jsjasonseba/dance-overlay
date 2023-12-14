import cv2
import random
import mediapipe as mp
import numpy as np

## initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def overlay_image (base_img, overlay_img, x_offset=0, y_offset=0, fill = False, optimize = False):
    # border_size = 400
    height, width, channels = base_img.shape
    if optimize == False:
        border_size = max(width, height)

    # overlay_img_size = (int)(width*0.85)

    base_img = cv2.copyMakeBorder(
        base_img,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 255)
    )

    x_offset += border_size
    y_offset += border_size
    

    # print(base_img.shape)
    if fill == False:
        overlay_img = cv2.resize(overlay_img, ((int)(width*0.85), (int)(width*0.59)))
    else:
        overlay_img = cv2.resize(overlay_img, (width, height))

    

    y1, y2 = y_offset - int(overlay_img.shape[0]/2), y_offset + int(overlay_img.shape[0]/2)
    x1, x2 = x_offset - int(overlay_img.shape[1]/2), x_offset + int(overlay_img.shape[1]/2)


    alpha_s = overlay_img[:, :, 3] / 255.0 if overlay_img.shape[2] == 4 else 1.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        base_img[y1:y2, x1:x2, c] = (alpha_s * overlay_img[:, :, c] +
                                  alpha_l * base_img[y1:y2, x1:x2, c])
        
    base_img = base_img[border_size:-border_size, border_size:-border_size]
    
    return base_img

class Animation():
    def __init__(self, prefix, offset_x, offset_y, frame = 2) -> None:
        self.prefix = prefix
        self.variant = 0
        self.fill = True
        if self.prefix == "animations/step1/step":
            self.variant = str(random.randint(0, 2))
            self.fill = False
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.current_iteration = 0
        self.current_frame = 0
        self.frame = frame
    
    def get_image(self):
        return cv2.imread(f"{self.prefix}{self.variant}{self.current_iteration}.png", cv2.IMREAD_UNCHANGED)

    def fetch_next_iteration(self):
        self.current_frame += 1
        if self.current_frame == self.frame:
            self.current_frame = 0
            self.current_iteration += 1
        return self.get_image()
    
    def apply_overlay_image(self, base_img):
        image = self.fetch_next_iteration()
        if image is not None:
            return overlay_image(base_img, image, self.offset_x, self.offset_y, self.fill)
        else:
            return None
        

animation_queue = []


# Load the dance video
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)
input_fps = cap.get(cv2.CAP_PROP_FPS)
overlay_fps = round(input_fps/15)
print(overlay_fps)

# Load the chart video
second_video_path = "input2.mp4"
cap2 = cv2.VideoCapture(second_video_path)
second_input_fps = cap2.get(cv2.CAP_PROP_FPS)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening dance video file.")
    exit()

# Check if the second video opened successfully
if not cap2.isOpened():
    print("Error opening chart video file.")
    exit()

# Create a VideoWriter object to save the output
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, input_fps, (360, 640))  # Adjust the resolution as needed

# Load the smaller image
small_image = cv2.imread("animations\step1\placeholder1.png", cv2.IMREAD_UNCHANGED)

# Process each frame in the video
while True:
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()

    # Break the loop if the video is finished
    if not ret:
        break

    # chart vid may be shorter or may not even exist
    if ret2:
        frame2 = cv2.resize(frame2, (604, 240))
        cv2.imshow("Chart Video", frame2)

    frame = cv2.resize(frame, (360, 640))
    
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    pose_results = pose.process(frame_rgb)

    next_queue = []
    for animation in animation_queue:
        image = animation.apply_overlay_image(frame)
        if image is not None:
            frame = image
            next_queue.append(animation)
    
    animation_queue = next_queue

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow("Overlay Result", frame)

    # Poll for keys
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if key == ord('d'): # left foot
        if pose_results.pose_landmarks:
            right_foot = pose_results.pose_landmarks.landmark[28] if pose_results.pose_landmarks.landmark[32].visibility > 0.1 else None
            h, w, _ = frame.shape
            right_foot_x, right_foot_y = int(right_foot.x * w), int(right_foot.y * h)
            
            animation_queue.append(Animation("animations/step1/step", right_foot_x - 20, right_foot_y - 25, overlay_fps))
    
    if key == ord('f'): # right foot
        if pose_results.pose_landmarks:
            left_foot = pose_results.pose_landmarks.landmark[27] if pose_results.pose_landmarks.landmark[31].visibility > 0.1 else None
            h, w, _ = frame.shape
            left_foot_x, left_foot_y = int(left_foot.x * w), int(left_foot.y * h)
            
            animation_queue.append(Animation("animations/step1/step", left_foot_x + 20, left_foot_y - 25, overlay_fps))
    
    if key == ord('k'): # jump
        xheight, xwidth, xchannels = frame.shape
        animation_queue.append(Animation("animations\jump\jump_0000", (int)(xwidth / 2),  (int)(xheight / 2), overlay_fps))

    if key == ord('j'): # down
        xheight, xwidth, xchannels = frame.shape
        animation_queue.append(Animation("animations\down\down0_0000", (int)(xwidth / 2),  (int)(xheight / 2), overlay_fps))

    # print(animation_queue)

# Release video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()