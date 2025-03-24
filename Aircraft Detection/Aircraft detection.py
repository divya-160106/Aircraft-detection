import cv2
import numpy as np
import argparse
import os
from scipy.spatial import distance as dist

def detect_airplane(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)
    return None

def calculate_speed(prev_center, curr_center, fps, scale_factor):
    if prev_center is None or curr_center is None:
        return 0
    distance = dist.euclidean(prev_center, curr_center)
    speed = (distance * scale_factor) * fps 
    return round(speed, 2)

def process_video(input_path, output_path, distance, scale_factor=0.1):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_path = "airplane.avi"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    prev_bbox = None
    future_path = []
    keypoints = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bbox = detect_airplane(frame)
        if bbox:
            x, y, w, h = bbox
            center = (x + w // 2, y + h // 2)
            prev_bbox = bbox if prev_bbox is None else prev_bbox
            speed = calculate_speed((prev_bbox[0], prev_bbox[1]), (x, y), fps, scale_factor)
            keypoints.append((x, y))  

            if len(keypoints) > 10:
                keypoints.pop(0)
            
            if len(keypoints) > 3:
                dx = np.mean([keypoints[i][0] - keypoints[i-1][0] for i in range(1, len(keypoints))])
                dy = np.mean([keypoints[i][1] - keypoints[i-1][1] for i in range(1, len(keypoints))])
                future_point = (int(keypoints[-1][0] + dx * 5), int(keypoints[-1][1] + dy * 5))
                future_path.append(future_point)
                
            if len(future_path) > 5:
                future_path.pop(0)

            if frame_count % 5 == 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

            z_value = round(distance / (h + 1), 2)
            cv2.putText(frame, f"X={x:.1f} Y={y:.1f} Z={z_value:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Speed: {speed} units/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if len(future_path) > 1:
                for i in range(1, len(future_path)):
                    cv2.line(frame, future_path[i - 1], future_path[i], (0, 0, 255), 2)
            
            prev_bbox = bbox
        
        frame_count += 1
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input WebM video")
    parser.add_argument("--output", type=str, required=True, help="Path to save output WebM video")
    parser.add_argument("--distance", type=float, required=True, help="Distance from viewer to airplane")
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.distance)
