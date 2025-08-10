import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from itertools import combinations
import colorsys # Added for distinct colors

# Initialize models
print("Loading models...")
try:
    # Load YOLOv8-Pose model with lower confidence threshold
    model = YOLO('yolov8n-pose.pt')
    print("✓ YOLOv8-Pose model loaded")
    
    # Load Mask R-CNN model
    maskrcnn = models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    maskrcnn.eval()
    print("✓ Mask R-CNN model loaded")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

def process_image(frame):
    """Process a single image with YOLOv8-Pose and Mask R-CNN"""
    results = []
    debug_frame = frame.copy()
    
    # Run YOLOv8-Pose with lower confidence threshold
    pose_results = model(frame, conf=0.25, verbose=False)  # Lowered from 0.3
    
    # Process each detection
    for r in pose_results:
        boxes = r.boxes
        keypoints = r.keypoints
        
        if len(boxes) == 0:
            print("No detections found. Trying with lower confidence...")
            # Try again with even lower confidence
            pose_results = model(frame, conf=0.15, verbose=False)  # Lowered from 0.2
            boxes = pose_results[0].boxes
            keypoints = pose_results[0].keypoints
        
        for i, (box, kpts) in enumerate(zip(boxes, keypoints)):
            if model.names[int(box.cls)] != 'person':
                continue
            
            # Get bounding box with padding
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Add padding to the box
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            # Debug: Draw bounding box
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get keypoints
            kpts_coords = kpts.data[0].cpu().numpy()
            print(f"Person {i} keypoints shape: {kpts_coords.shape}")
            
            # Get keypoint coordinates and confidences
            if kpts_coords.shape[0] >= 17:  # COCO format has 17 keypoints
                nose = kpts_coords[0]
                left_ankle = kpts_coords[15]
                right_ankle = kpts_coords[16]
                
                # Debug: Print keypoint confidences
                print(f"Nose confidence: {nose[2]:.2f}")
                print(f"Left ankle confidence: {left_ankle[2]:.2f}")
                print(f"Right ankle confidence: {right_ankle[2]:.2f}")
                
                # Draw keypoints on debug frame
                for kpt in [nose, left_ankle, right_ankle]:
                    if kpt[2] > 0.3:  # If confidence is above threshold
                        cv2.circle(debug_frame, (int(kpt[0]), int(kpt[1])), 4, (0, 255, 255), -1)
            
            # Crop and process with Mask R-CNN
            cropped_person = frame[y1:y2, x1:x2]
            if cropped_person.size == 0:
                print(f"Invalid crop for person {i}")
                continue
            
            # Convert to PIL and prepare for Mask R-CNN
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB))
            transform = transforms.Compose([transforms.ToTensor()])
            img_tensor = transform(cropped_pil).unsqueeze(0)
            
            try:
                # Run Mask R-CNN with error handling
                with torch.no_grad():
                    prediction = maskrcnn(img_tensor)[0]
                
                # Get person mask with confidence threshold
                person_indices = [j for j, (label, score) in enumerate(zip(prediction['labels'], prediction['scores'])) 
                                if label == 1 and score > 0.7]
                
                if not person_indices:
                    print(f"No confident mask for person {i}")
                    continue
                
                # Get the most confident mask
                best_mask_idx = person_indices[0]
                mask = prediction['masks'][best_mask_idx, 0].cpu().numpy()
                binary_mask = (mask > 0.5).astype(np.uint8)
                
                # Debug: Draw mask outline
                mask_display = np.zeros_like(cropped_person)
                mask_display[:,:,1] = binary_mask * 255
                debug_frame[y1:y2, x1:x2] = cv2.addWeighted(
                    debug_frame[y1:y2, x1:x2], 1, mask_display, 0.3, 0)
                
                # Define search regions
                bbox_height = y2 - y1
                head_region_height = int(bbox_height * 0.15)  # Increased from 0.05
                foot_region_height = int(bbox_height * 0.15)  # Increased from 0.10
                
                # Find head point
                head_region = binary_mask[:head_region_height, :]
                head_y_local = None
                if head_region.any():
                    head_ys, head_xs = np.where(head_region == 1)
                    if len(head_ys) > 0:
                        head_y_local = head_ys.min()
                        head_x_local = head_xs[head_ys.argmin()]
                        head_point = (head_x_local + x1, head_y_local + y1)
                        cv2.circle(debug_frame, head_point, 5, (0, 255, 0), -1)
                
                # Find foot point
                foot_region = binary_mask[-foot_region_height:, :]
                foot_y_local = None
                if foot_region.any():
                    foot_ys, foot_xs = np.where(foot_region == 1)
                    if len(foot_ys) > 0:
                        foot_y_local = foot_ys.max() + (binary_mask.shape[0] - foot_region.shape[0])
                        foot_x_local = foot_xs[foot_ys.argmax()]
                        foot_point = (foot_x_local + x1, foot_y_local + y1)
                        cv2.circle(debug_frame, foot_point, 5, (0, 0, 255), -1)
                
                if head_y_local is not None and foot_y_local is not None:
                    # Draw head-foot line
                    cv2.line(debug_frame, head_point, foot_point, (255, 0, 0), 2)
                    
                    results.append({
                        'head': head_point,
                        'foot': foot_point,
                        'bbox': (x1, y1, x2, y2),
                        'height_pixels': foot_point[1] - head_point[1]
                    })
                    
                    # Add height text
                    height_pixels = foot_point[1] - head_point[1]
                    cv2.putText(debug_frame, f'Height: {height_pixels}px', 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
            except Exception as e:
                print(f"Error processing mask for person {i}: {e}")
                continue
    
    if not results:
        print("No valid detections found in the image")
    
    return debug_frame, results

input_type = input("Enter input type ('video' or 'photo'): ").strip().lower()

from collections import defaultdict
id_to_headfoot = defaultdict(list)
processed_ids = set()


if input_type == 'video':
    video_path = input("Enter the path to the video file: ").strip()
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            frames.append(frame)
        else:
            break
    cap.release() #closes the video file after all frames are read.

# If there are valid frames, it loops through each one.
    if frames:
        print(f"Processing {len(frames)} frames with tracking...")
        for frame_idx, frame in enumerate(frames):
            # Run YOLOv8 with tracking (ByteTrack)
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            # Iterates through each detection in the frame.
            for r in results:
                for i, box in enumerate(r.boxes):
                    if model.names[int(box.cls)] == 'person':
                        track_id = int(box.id) if hasattr(box, 'id') and box.id is not None else None
                        if track_id is None or track_id in processed_ids:
                            continue  # skip if no ID or already processed
                        # --- Only process this ID the first time ---
                        processed_ids.add(track_id)
                        # Extract bounding box coordinates (x1, y1, x2, y2) and crop the person from the frame. 
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cropped_person = frame[y1:y2, x1:x2]
                        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)) # Converts OpenCV BGR image to a PIL RGB image (needed for PyTorch).
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                        ])
                        img_tensor = transform(cropped_pil).unsqueeze(0)
                        with torch.no_grad():
                            prediction = maskrcnn(img_tensor)[0]
                        person_indices = [j for j, label in enumerate(prediction['labels']) if label == 1]
                        # If there is a person detected, it extracts the mask and finds the head and foot positions.
                        if person_indices:
                            mask = prediction['masks'][person_indices[0], 0].cpu().numpy()
                            binary_mask = (mask > 0.5).astype(np.uint8)
                            ys, xs = np.where(binary_mask == 1)
                            if len(ys) > 0:
                                h2D_y = ys.min()
                                h2D_x = xs[ys.argmin()]
                                f2D_y = ys.max()
                                f2D_x = xs[ys.argmax()]
                                # Offset by crop position
                                h2D = (h2D_x + x1, h2D_y + y1)
                                f2D = (f2D_x + x1, f2D_y + y1)
                                id_to_headfoot[track_id].append((h2D[0], h2D[1], f2D[0], f2D[1]))
                                # Optionally save masks for visualization
                                np.save(f'binary_mask_{frame_idx}_{track_id}.npy', binary_mask)
                                cv2.imwrite(f'person_mask_{frame_idx}_{track_id}.png', binary_mask * 255)
                                cv2.imwrite(f'cropped_person_{frame_idx}_{track_id}.jpg', cropped_person)
                                print(f"Frame {frame_idx}, ID {track_id}: Head {h2D}, Foot {f2D}")
                            else:
                                print(f'Frame {frame_idx}, ID {track_id}: No mask pixels found.')
                        else:
                            print(f'Frame {frame_idx}, ID {track_id}: No person detected by Mask R-CNN.')
    else:
        print("No frames found in video.")
        exit()
elif input_type == 'photo':
    # Ask for at least 2 photo paths, separated by commas
    photo_paths = input("Enter the paths to at least 2 photos, separated by commas: ").strip().split(',')
    photo_paths = [p.strip() for p in photo_paths if p.strip()]
    if len(photo_paths) < 2:
        print("Please provide at least 2 photo paths.")
        exit()
    image_head_foot_pairs = []  # Store head-foot pairs per image
    for photo_idx, photo_path in enumerate(photo_paths):
        selected_frame = cv2.imread(photo_path)
        if selected_frame is None:
            print(f"Could not read the image: {photo_path}")
            image_head_foot_pairs.append([])
            continue
        processed_frame, detection_results = process_image(selected_frame)
        pairs = [(d['head'][0], d['head'][1], d['foot'][0], d['foot'][1]) for d in detection_results]
        # Print head and foot positions for each detected person
        for person_idx, (hx, hy, fx, fy) in enumerate(pairs):
            print(f"Image: {photo_path}, Person {person_idx}: Head 2D position: ({hx}, {hy}) and Foot 2D position: ({fx}, {fy})")
        image_head_foot_pairs.append(pairs)
        
        # Optionally save or display processed_frame
        cv2.imshow(f"Processed Image: {photo_path}", processed_frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    # After collecting all head-foot pairs, proceed to VP estimation and visualization for each image
    # Step 2: Convert head-foot points to line equations (Ax + By + C = 0)
    for idx, (photo_path, pairs) in enumerate(zip(photo_paths, image_head_foot_pairs)):
        print(f"\nProcessing image {idx}: {photo_path}")
        
        if len(pairs) < 2:  # Need at least 2 people for VP estimation
            print(f"Skipping image {idx}: Not enough people detected (minimum 2 required)")
            continue
        
        # Process each image separately
        lines = []
        # Convert each head-foot pair to a line equation for THIS IMAGE ONLY
        for (hx, hy, fx, fy) in pairs:
            A = fy - hy
            B = hx - fx
            C = fx * hy - hx * fy
            norm = np.sqrt(A*A + B*B)  # Fixed the squared terms
            if norm > 1e-6:
                lines.append((A/norm, B/norm, C/norm))
                
        print(f"Number of valid lines in image {idx}: {len(lines)}")
        
        # Step 3: Applying RANSAC to find the vertical vanishing point for THIS IMAGE
        best_vp = None
        max_inliers = 0
        threshold = 5  # pixel distance threshold
        iterations = 100000
        
        # Store the VVP when it's computed during vertical vanishing point estimation
        if len(lines) >= 2:
            for _ in range(iterations):
                # Randomly select 2 lines from THIS IMAGE ONLY
                idx1, idx2 = np.random.choice(len(lines), 2, replace=False)
                l1, l2 = lines[idx1], lines[idx2]
                
                # Compute intersection
                d = l1[0]*l2[1] - l2[0]*l1[1]
                if abs(d) < 1e-6:
                    continue  # Skip parallel lines
                
                x = (l1[1]*l2[2] - l2[1]*l1[2]) / d
                y = (l2[0]*l1[2] - l1[0]*l2[2]) / d
                
                # Count inliers from THIS IMAGE ONLY
                inliers = 0
                for A, B, C in lines:
                    dist = abs(A*x + B*y + C)
                    if dist < threshold:
                        inliers += 1
                        
                if inliers > max_inliers:
                    max_inliers = inliers
                    best_vp = (int(round(x)), int(round(y)))
            
            if best_vp:
                print(f"Image {idx} - Estimated vertical vanishing point: {best_vp}")
                print(f"Image {idx} - Number of inliers: {max_inliers} out of {len(lines)} lines")
                
                # Store VVP for this image
                image_vvp = best_vp  # Store VVP for this specific image
                
                # Visualization for THIS IMAGE
                selected_frame = cv2.imread(photo_path)
                if selected_frame is None:
                
                    continue
                    
                vis_img = selected_frame.copy()
                
                # Draw original head-foot lines in green
                for (hx, hy, fx, fy) in pairs:
                
                    cv2.line(vis_img, (hx, hy), (fx, fy), (0, 255, 0), 2)
                    # Draw lines to vanishing point in blue
                    cv2.line(vis_img, (hx, hy), best_vp, (255, 0, 0), 1)
                    cv2.line(vis_img, (fx, fy), best_vp, (255, 0, 0), 1)
                
                    
                # Draw vanishing point in red
                cv2.circle(vis_img, best_vp, 8, (0, 0, 255), -1)
                cv2.putText(vis_img, 'VVP', (best_vp[0]+10, best_vp[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
                out_path = f'vertical_vanishing_point_image_{idx}.png'
                cv2.imwrite(out_path, vis_img)
                print(f"Image {idx} - Vanishing point overlay saved as {out_path}")
                
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
                plt.title(f'Vertical Vanishing Point - Image {idx}')
                plt.axis('off')
                plt.show()
            else:
                
                print(f"Image {idx} - Could not estimate a vertical vanishing point")
        else:
            print(f"Image {idx} - Not enough lines for vanishing point estimation")
    
    


            
    # --- Horizontal Vanishing Point Estimation (Head-to-Head and Foot-to-Foot) ---

    for idx, (photo_path, pairs) in enumerate(zip(photo_paths, image_head_foot_pairs)):
        
        print(f"\nProcessing horizontal vanishing points for image {idx}: {photo_path}")
    
        selected_frame = cv2.imread(photo_path)
        if selected_frame is None:
            
            print(f"Skipping image {idx}: Invalid image")
            
            continue
        
    # Need exactly 3 people for this approach
        if len(pairs) != 3:
            
            print(f"Skipping image {idx}: Need exactly 3 people (found {len(pairs)})")
            continue
        
    vis_img = selected_frame.copy()
    h, w = selected_frame.shape[:2]
    
    # Sort people by x-coordinate to get left, middle, right
    sorted_pairs = sorted(enumerate(pairs), key=lambda x: x[1][0])  # Sort by head x-coordinate
    left_idx, left_pair = sorted_pairs[0]
    middle_idx, middle_pair = sorted_pairs[1]
    right_idx, right_pair = sorted_pairs[2]
    
    print(f"\nPeople positions in image {idx}:")
    print(f"Left person (idx {left_idx}): Head at ({left_pair[0]}, {left_pair[1]}), Foot at ({left_pair[2]}, {left_pair[3]})")
    print(f"Middle person (idx {middle_idx}): Head at ({middle_pair[0]}, {middle_pair[1]}), Foot at ({middle_pair[2]}, {middle_pair[3]})")
    print(f"Right person (idx {right_idx}): Head at ({right_pair[0]}, {right_pair[1]}), Foot at ({right_pair[2]}, {right_pair[3]})")
    
    def extend_line(p1, p2, factor=2000):
        """Extend a line segment by a factor"""
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        # Extend in both directions
        new_x1 = x1 - factor * dx
        new_y1 = y1 - factor * dy
        new_x2 = x2 + factor * dx
        new_y2 = y2 + factor * dy
        return (int(new_x1), int(new_y1)), (int(new_x2), int(new_y2))
    
    def compute_intersection(line1_p1, line1_p2, line2_p1, line2_p2):
        """Compute the intersection point of two lines"""
        x1, y1 = line1_p1
        x2, y2 = line1_p2
        x3, y3 = line2_p1
        x4, y4 = line2_p2
        
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-6:
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (int(round(x)), int(round(y)))
    
    # Create a larger visualization image
    margin_w = w  # Add space on sides
    vis_extended = np.zeros((h, w + 2*margin_w, 3), dtype=np.uint8)
    vis_extended[:, margin_w:margin_w+w] = vis_img
    
    # Colors for visualization
    LEFT_COLOR = (0, 255, 0)    # Green
    RIGHT_COLOR = (0, 0, 255)   # Red
    MIDDLE_COLOR = (255, 255, 0) # Cyan
    
    # Draw all detected people
    for person_idx, (hx, hy, fx, fy) in [(left_idx, left_pair), (middle_idx, middle_pair), (right_idx, right_pair)]:
        
        color = MIDDLE_COLOR if person_idx == middle_idx else (LEFT_COLOR if person_idx == left_idx else RIGHT_COLOR)
        # Draw head and foot points
        cv2.circle(vis_extended, (hx + margin_w, hy), 6, color, -1)
        cv2.circle(vis_extended, (fx + margin_w, fy), 6, color, -1)
        # Draw vertical line
        cv2.line(vis_extended, (hx + margin_w, hy), (fx + margin_w, fy), color, 2)
        # Add labels
        label = "Middle" if person_idx == middle_idx else ("Left" if person_idx == left_idx else "Right")
        cv2.putText(vis_extended, label, (hx + margin_w - 30, hy - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Compute and draw left HVP (middle to left person)
    # Head-to-head line
    left_head_ext = extend_line((middle_pair[0], middle_pair[1]), (left_pair[0], left_pair[1]))
    cv2.line(vis_extended, 
            (middle_pair[0] + margin_w, middle_pair[1]),
            (left_pair[0] + margin_w, left_pair[1]),
            LEFT_COLOR, 2)
    cv2.line(vis_extended,
            (left_head_ext[0][0] + margin_w, left_head_ext[0][1]),
            (left_head_ext[1][0] + margin_w, left_head_ext[1][1]),
            LEFT_COLOR, 1)
    
    # Foot-to-foot line
    left_foot_ext = extend_line((middle_pair[2], middle_pair[3]), (left_pair[2], left_pair[3]))
    cv2.line(vis_extended,
            (middle_pair[2] + margin_w, middle_pair[3]),
            (left_pair[2] + margin_w, left_pair[3]),
            LEFT_COLOR, 2)
    cv2.line(vis_extended,
            (left_foot_ext[0][0] + margin_w, left_foot_ext[0][1]),
            (left_foot_ext[1][0] + margin_w, left_foot_ext[1][1]),
            LEFT_COLOR, 1)
    
    # Compute left HVP
    left_hvp = compute_intersection(
        left_head_ext[0], left_head_ext[1],
        left_foot_ext[0], left_foot_ext[1]
    )
    
    # Compute and draw right HVP (middle to right person)
    # Head-to-head line
    right_head_ext = extend_line((middle_pair[0], middle_pair[1]), (right_pair[0], right_pair[1]))
    cv2.line(vis_extended,
            (middle_pair[0] + margin_w, middle_pair[1]),
            (right_pair[0] + margin_w, right_pair[1]),
            RIGHT_COLOR, 2)
    cv2.line(vis_extended,
            (right_head_ext[0][0] + margin_w, right_head_ext[0][1]),
            (right_head_ext[1][0] + margin_w, right_head_ext[1][1]),
            RIGHT_COLOR, 1)
    
    # Foot-to-foot line
    right_foot_ext = extend_line((middle_pair[2], middle_pair[3]), (right_pair[2], right_pair[3]))
    cv2.line(vis_extended,
            (middle_pair[2] + margin_w, middle_pair[3]),
            (right_pair[2] + margin_w, right_pair[3]),
            RIGHT_COLOR, 2)
    cv2.line(vis_extended,
            (right_foot_ext[0][0] + margin_w, right_foot_ext[0][1]),
            (right_foot_ext[1][0] + margin_w, right_foot_ext[1][1]),
            RIGHT_COLOR, 1)
    
    # Compute right HVP
    right_hvp = compute_intersection(
        right_head_ext[0], right_head_ext[1],
        right_foot_ext[0], right_foot_ext[1]
    )
    
    # Draw vanishing points if found
    if left_hvp:
        left_hvp_adj = (left_hvp[0] + margin_w, left_hvp[1])
        cv2.circle(vis_extended, left_hvp_adj, 10, LEFT_COLOR, -1)
        cv2.putText(vis_extended, 'Left HVP', (left_hvp_adj[0]-100, left_hvp_adj[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, LEFT_COLOR, 2)
        print(f"Left HVP found at: {left_hvp}")
    else:
        print("Could not compute left HVP")
        
    if right_hvp:
        right_hvp_adj = (right_hvp[0] + margin_w, right_hvp[1])
        cv2.circle(vis_extended, right_hvp_adj, 10, RIGHT_COLOR, -1)
        cv2.putText(vis_extended, 'Right HVP', (right_hvp_adj[0]+10, right_hvp_adj[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, RIGHT_COLOR, 2)
        print(f"Right HVP found at: {right_hvp}")
    else:
        print("Could not compute right HVP")
    
    # Draw horizon line if both VPs are found
    if left_hvp and right_hvp:
        # Draw the horizon line
        cv2.line(vis_extended,
                (left_hvp[0] + margin_w, left_hvp[1]),
                (right_hvp[0] + margin_w, right_hvp[1]),
                (0, 255, 255), 2)  # Yellow horizon line
        
        # Compute the vanishing line equation (a1x + a2y + a3 = 0)
        # Using two points form: (y - y1) = m(x - x1)
        x1, y1 = left_hvp
        x2, y2 = right_hvp
        
        if abs(x2 - x1) > 1e-6:  # Check if line is not vertical
            # Calculate slope
            m = (y2 - y1) / (x2 - x1)
            
            # Convert to general form: a1x + a2y + a3 = 0
            a1 = m  # coefficient of x
            a2 = -1  # coefficient of y (normalized)
            a3 = y1 - m*x1  # constant term
            
            # Normalize coefficients
            norm = np.sqrt(a1*a1 + a2*a2)
            a1 = a1 / norm
            a2 = a2 / norm
            a3 = a3 / norm
            
            # Store the vanishing line parameters
            vanishing_line = (a1, a2, a3)
            
            # Print the equation
            print(f"\nVanishing Line Equation (Horizon):")
            print(f"{a1:.3f}x + {a2:.3f}y + {a3:.3f} = 0")
            
            # Add equation to visualization
            mid_x = (x1 + x2) // 2
            mid_y = int((-a3 - a1*mid_x) / a2)
            equation_text = f"Horizon: {a1:.2f}x + {a2:.2f}y + {a3:.2f} = 0"
            cv2.putText(vis_extended, equation_text,
                      (int(mid_x) + margin_w, int(mid_y) - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            print("\nKey parameters for focal length computation:")
            print(f"a2 (coefficient of y): {a2:.3f}")
            print(f"a3 (constant term): {a3:.3f}")
            print(f"py (principal point y): {h/2:.1f}")
            if best_vp:
                print(f"vy (vertical VP y): {best_vp[1]:.1f}")
        else:
            print("Cannot compute vanishing line equation: HVPs are vertically aligned")
            vanishing_line = None
    else:
        print("Cannot compute vanishing line: Missing one or both horizontal vanishing points")
        vanishing_line = None
        
        

    

    # --- Camera Calibration using Vanishing Points ---
    if left_hvp and right_hvp and image_vvp:  # Use image_vvp instead of best_vp
        print("\nComputing camera calibration parameters...")
        print(f"Using VVP from current image {idx}: {image_vvp}")
        print(f"Left HVP from image: {idx} : {left_hvp}")
        print(f"Right HVP from image: {idx} : {right_hvp}")
        
        
        # Get image dimensions and principal point
        h, w = selected_frame.shape[:2]
        
        # Get EXTENDED image dimensions for principal point calculation
        h, w = selected_frame.shape[:2]  # Original dimensions
        extended_w = w + 2 * margin_w    # Extended canvas width
        extended_h = h                   # Height unchanged
        # Calculate principal point for EXTENDED canvas
        px = extended_w / 2  # Principal point x (extended canvas center)
        py = extended_h / 2  # Principal point y (same as original)
    
        print(f"Original image dimensions: {w} × {h}")
        print(f"Extended canvas dimensions: {extended_w} × {extended_h}")
        print(f"Principal point (extended): ({px}, {py})")
    
        # **INSERT THE COORDINATE CONVERSION HERE**
        # Convert HVPs from extended to original coordinates
        '''left_hvp_original = (left_hvp[0] - margin_w, left_hvp[1]) if left_hvp else None
        right_hvp_original = (right_hvp[0] - margin_w, right_hvp[1]) if right_hvp else None
        
        print(f"\nCoordinate conversion:")
        print(f"Left HVP - Extended: {left_hvp}, Original: {left_hvp_original}")
        print(f"Right HVP - Extended: {right_hvp}, Original: {right_hvp_original}")
        print(f"Margin offset applied: {margin_w} pixels")  '''
        #image_vvp[1] = image_vvp_filtered
        # 1. Express vanishing points in homogeneous coordinates
        v1 = np.array([left_hvp[0], left_hvp[1], 1], dtype=np.float64)    # Left HVP
        v2 = np.array([right_hvp[0], right_hvp[1], 1], dtype=np.float64)  # Right HVP
        v3 = np.array([image_vvp[0], image_vvp[1], 1], dtype=np.float64)  # Vertical VP'''
        '''v1 = np.array([left_hvp_original[0], left_hvp_original[1], 1], dtype=np.float64)    # Left HVP
        v2 = np.array([right_hvp_original[0], right_hvp_original[1], 1], dtype=np.float64)  # Right HVP
        v3 = np.array([image_vvp[0], image_vvp[1], 1], dtype=np.float64)  # Vertical VP (already in original coords)'''
        # Normalize vanishing points before cross product
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        print("\nVanishing Points (normalized):")
        print(f"v1 (Left HVP)   : {v1}")
        print(f"v2 (Right HVP)  : {v2}")
        print(f"v3 (VVP)        : {v3} (from same image {idx})")
        
        # Verify the vanishing points are from the same image
        print("\nVerifying vanishing points from image {idx}:")
        print(f"Left HVP coordinates : ({left_hvp[0]}, {left_hvp[1]})")
        print(f"Right HVP coordinates: ({right_hvp[0]}, {right_hvp[1]})")
        print(f"VVP coordinates      : ({image_vvp[0]}, {image_vvp[1]})")
        
        # 2. Compute vanishing line from HVPs
        vanishing_line = np.cross(v1, v2)
        a1, a2, a3 = vanishing_line
        
        # Normalize the line coefficients
        norm = np.sqrt(a1*a1 + a2*a2)
        if norm > 1e-6:
            a1 = a1 / norm
            a2 = a2 / norm
            a3 = a3 / norm
            
            print("\nVanishing Line Equation:")
            print(f"{a1:.3f}x + {a2:.3f}y + {a3:.3f} = 0")
            
            # 3. Compute focal length using VVP from same image
            vy = image_vvp[1]  # y-coordinate of vertical vanishing point
            if abs(a2) > 1e-6:
                f = np.sqrt(abs((a3/a2 - py) * (vy - py)))
                print(f"\nComputed focal length: {f:.2f}")
                
                # 4. Construct calibration matrix K
                K = np.array([
                    [f, 0, px],
                    [0, f, py],
                    [0, 0, 1]
                ])
                print("\nCalibration Matrix K:")
                print(K)
                
                # 5. Compute rotation matrix R
                K_inv = np.linalg.inv(K)
                
                # Compute camera coordinate directions using VPs from same image
                r1_prime = K_inv @ v1  # X direction (from left HVP)
                r2_prime = K_inv @ v2  # Y direction (from right HVP)
                r3_prime = K_inv @ v3  # Z direction (from VVP of same image)
                
                # Normalize direction vectors
                r1 = r1_prime / np.linalg.norm(r1_prime)
                r2 = r2_prime / np.linalg.norm(r2_prime)
                r3 = r3_prime / np.linalg.norm(r3_prime)
                
                # Stack vectors to form rotation matrix
                R = np.column_stack((r1, r2, r3))
                
                # Ensure R is a valid rotation matrix using SVD
                U, S, Vh = np.linalg.svd(R)
                
                # Fix the determinant issue by ensuring proper rotation
                det_U = np.linalg.det(U)
                det_Vh = np.linalg.det(Vh)
                
                # If either determinant is negative, adjust the last column/row
                if det_U < 0:
                    U[:, -1] *= -1
                if det_Vh < 0:
                    Vh[-1, :] *= -1
                
                # Reconstruct R with proper rotation
                R = U @ Vh
                
                # Verify R is orthonormal
                det_R = np.linalg.det(R)
                print(f"\nDeterminant of R: {det_R:.6f} (should be ≈ 1)")
                
                if abs(det_R - 1.0) > 1e-6:
                    print("Warning: R may not be a proper rotation matrix")
                else:
                    print("R is a proper rotation matrix (det ≈ 1)")
                
                    # Check column norms of R
                col_norms = [np.linalg.norm(R[:, i]) for i in range(3)]
                print(f"R column norms: {col_norms} (should all be ≈ 1.0)")
                
                # Get camera height from user
                camera_height = float(input("\nEnter the camera height from ground in meters: ").strip())
                print(f"Camera height: {camera_height} meters")
                
                # 6. Compute translation vector t using camera height
                t = np.array([[0],                # X translation (assumed 0)
                            [0],                # Y translation (assumed 0)
                            [-camera_height]])  # Z translation (negative because camera looks down)
                
                print("\nTranslation vector t (in meters):")
                print(t)
                
                # 7. Construct full projection matrix P = K[R|t]
                Rt = np.hstack((R, t))
                P = K @ Rt
                
                # Normalize P
                #P = P / P[2,3]
                
                print("\nFull Projection Matrix P = K[R|t]:")
                print(P)
                
                # Save matrices for later use
                np.save(f'calibration_K_{idx}.npy', K)
                np.save(f'rotation_R_{idx}.npy', R)
                np.save(f'projection_P_{idx}.npy', P)
                
                # Add calibration info to visualization
                info_text = [
                    f"f = {f:.1f}",
                    f"px, py = ({px:.1f}, {py:.1f})",
                    f"det(R) = {det_R:.3f}",
                    f"Camera height = {camera_height:.2f}m",
                    f"VVP = ({image_vvp[0]:.1f}, {image_vvp[1]:.1f})"
                ]
                y_offset = 30
                for text in info_text:
                    cv2.putText(vis_extended, text,
                              (20, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y_offset += 30
            else:
                print("Error: Invalid vanishing line (a2 ≈ 0)")
        else:
            print("Error: Invalid vanishing line normalization")
    else:
        print("Cannot compute calibration: Missing vanishing points")
        
        
        
    ##-- FUNCTIONS FOR VALIDATION OF PARAMETERS OF P MATRIX ---#
    def validate_focal_length(f, image_width, image_height):
        
        """Validate if the computed focal length is reasonable"""
        print(f"=== FOCAL LENGTH VALIDATION ===")
        print(f"Computed focal length: {f:.2f} pixels")
        # Check if focal length is in reasonable range
        diagonal_pixels = np.sqrt(image_width**2 + image_height**2)
        f_min = diagonal_pixels * 0.5  # Minimum reasonable focal length
        f_max = diagonal_pixels * 3.0  # Maximum reasonable focal length
    
        print(f"Image diagonal: {diagonal_pixels:.1f} pixels")
        print(f"Reasonable f range: [{f_min:.1f}, {f_max:.1f}]")
    
        is_valid = f_min <= f <= f_max
        print(f"Focal length validity: {'✓' if is_valid else 'W'}")
    
    # Check for NaN or infinity
        if np.isnan(f) or np.isinf(f):
            
            print("ERROR: Focal length is NaN or infinite")
            return False
    
        return is_valid
      
      
    def validate_intrinsic_matrix(K, image_width, image_height):
        
        """Comprehensive validation of intrinsic matrix K"""
        print(f"\n=== INTRINSIC MATRIX K VALIDATION ===")
        print("K matrix:")
        print(K)
    
    # Extract parameters
        fx, fy = K[0,0], K[1,1]    
        cx, cy = K[0,2], K[1,2]    
        skew = K[0,1]
    
        print(f"Focal lengths: fx={fx:.2f}, fy={fy:.2f}")
        print(f"Principal point: ({cx:.1f}, {cy:.1f})")
        print(f"Skew: {skew:.6f}")
    
    # Validation checks
        checks = {}
    
        # 1. Positive focal lengths
        checks['positive_focal'] = fx > 0 and fy > 0
        print(f"Positive focal lengths: {'✓' if checks['positive_focal'] else 'W'}")
    
        # 2. Equal focal lengths (for square pixels)
        checks['square_pixels'] = abs(fx - fy) / fx < 0.01
        print(f"Square pixels (fx≈fy): {'✓' if checks['square_pixels'] else 'W'}")
    
        # 3. Reasonable principal point location
        checks['principal_point'] = (0 < cx < image_width) and (0 < cy < image_height)
        print(f"Principal point in image bounds: {'✓' if checks['principal_point'] else 'W'}")
    
        # 4. Near-zero skew
        checks['zero_skew'] = abs(skew) < 1e-6
        print(f"Near-zero skew: {'✓' if checks['zero_skew'] else 'W'}")
    
        # 5. Matrix properties
        det_K = np.linalg.det(K)
        checks['positive_det'] = det_K > 0
        print(f"Positive determinant: {det_K:.2e} {'✓' if checks['positive_det'] else 'W'}")
    
        # 6. No NaN or infinity
        checks['finite_values'] = np.all(np.isfinite(K))
        print(f"All finite values: {'✓' if checks['finite_values'] else 'W'}")
    
        all_valid = all(checks.values())
        print(f"\nOverall K validation: {'✓ PASSED' if all_valid else 'W FAILED'}")
    
        return all_valid, checks

    def validate_rotation_matrix(R):
        
        """Comprehensive validation of rotation matrix R"""
        print(f"\n=== ROTATION MATRIX R VALIDATION ===")
        print("R matrix:")
        print(R)
    
        checks = {}
    
        # 1. Check determinant = +1
        det_R = np.linalg.det(R)
        checks['det_one'] = abs(det_R - 1.0) < 1e-6
        print(f"Determinant: {det_R:.8f} {'✓' if checks['det_one'] else 'W'}")
    
        # 2. Check orthogonality (R^T * R = I)
        RTR = R.T @ R
        identity_error = np.linalg.norm(RTR - np.eye(3))
        checks['orthogonal'] = identity_error < 1e-6
        print(f"Orthogonality error: {identity_error:.2e} {'✓' if checks['orthogonal'] else 'W'}")
    
        if not checks['orthogonal']:
                
            print("R^T * R =")
            print(RTR)
    
        # 3. Check column norms = 1
        col_norms = [np.linalg.norm(R[:, i]) for i in range(3)]
        norm_errors = [abs(norm - 1.0) for norm in col_norms]
        checks['unit_columns'] = all(error < 1e-6 for error in norm_errors)
        print(f"Column norms: {col_norms} {'✓' if checks['unit_columns'] else 'W'}")
    
        # 4. Check for NaN or infinity
        checks['finite_values'] = np.all(np.isfinite(R))
        print(f"All finite values: {'✓' if checks['finite_values'] else 'W'}")
    
        # 5. Condition number
        cond_R = np.linalg.cond(R)
        checks['well_conditioned'] = cond_R < 100
        print(f"Condition number: {cond_R:.2e} {'✓' if checks['well_conditioned'] else 'W'}")
    
        all_valid = all(checks.values())
        print(f"\nOverall R validation: {'✓ PASSED' if all_valid else 'W FAILED'}")
    
        # If R is invalid, provide correction
        if not all_valid:
            
            print("\nCorrecting R using SVD...")
            U, s, Vt = np.linalg.svd(R)
            R_corrected = U @ Vt
            if np.linalg.det(R_corrected) < 0:
                #
                Vt[-1, :] *= -1
                R_corrected = U @ Vt
            print("Corrected R:")
            print(R_corrected)
            return all_valid, checks, R_corrected
    
        return all_valid, checks, R

    def validate_projection_matrix(P, K, R, t):
        """Comprehensive validation of projection matrix P"""
        print(f"\n=== PROJECTION MATRIX P VALIDATION ===")
        print("P matrix:")
        print(P)
    
        checks = {}
    
        # 1. Check construction P = K[R|t]
        Rt = np.hstack([R, t.reshape(-1, 1)])
        P_expected = K @ Rt
        construction_error = np.linalg.norm(P - P_expected)
        checks['correct_construction'] = construction_error < 1e-10
        print(f"Construction error ||P - K[R|t]||: {construction_error:.2e} {'✓' if checks['correct_construction'] else 'W'}")
    
        if not checks['correct_construction']:
            #
            print("Expected P = K[R|t]:")
            print(P_expected)    
    
        # 2. Check matrix dimensions
        checks['correct_shape'] = P.shape == (3, 4)
        print(f"Correct shape (3×4): {'✓' if checks['correct_shape'] else 'W'}")
    
        # 3. Check for finite values
        checks['finite_values'] = np.all(np.isfinite(P))
        print(f"All finite values: {'✓' if checks['finite_values'] else 'W'}")
    
        # 4. Check condition number
        cond_P = np.linalg.cond(P)
        checks['well_conditioned'] = cond_P < 1e12
        print(f"P condition number: {cond_P:.2e} {'✓' if checks['well_conditioned'] else 'W'}")
    
        # 5. Check P^T P invertibility (critical for f3D computation)
        PTP = P.T @ P
        cond_PTP = np.linalg.cond(PTP)
        checks['PTP_invertible'] = cond_PTP < 1e12
        print(f"P^T P condition number: {cond_PTP:.2e} {'✓' if checks['PTP_invertible'] else 'W'}")
    
        # 6. Check last row structure
        last_row = P[2, :]
        print(f"Last row of P: {last_row}")
    
    # The last element should typically not be normalized to 1
        if abs(last_row[3] - 1.0) < 1e-6:
            #
            print("WARNING: P appears to be incorrectly normalized by P[2,3]")
            checks['normalization'] = False
        else:
            #
            checks['normalization'] = True
    
        all_valid = all(checks.values())
        print(f"\nOverall P validation: {'✓ PASSED' if all_valid else 'W FAILED'}")
    
        # Return corrected P if needed
        if not checks['correct_construction']:
            
            P_corrected = P_expected
            return all_valid, checks, P_corrected
    
        return all_valid, checks, P

    def complete_calibration_validation(f, K, R, t, P, image_width, image_height):
        """Run complete validation pipeline"""
        print("=" * 60)
        print("COMPLETE CAMERA CALIBRATION VALIDATION")
        print("=" * 60)
    
        #Validate each component
        f_valid = validate_focal_length(f, image_width, image_height)
        K_valid, K_checks = validate_intrinsic_matrix(K, image_width, image_height)
        R_valid, R_checks, R_corrected = validate_rotation_matrix(R)
        P_valid, P_checks, P_corrected = validate_projection_matrix(P, K, R_corrected, t)
    
        # Overall assessment
        all_components_valid = f_valid and K_valid and R_valid and P_valid
    
        print(f"\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Focal length (f): {'✓' if f_valid else 'W'}")
        print(f"Intrinsic matrix (K): {'✓' if K_valid else 'W'}")
        print(f"Rotation matrix (R): {'✓' if R_valid else 'W'}")
        print(f"Projection matrix (P): {'✓' if P_valid else 'W'}")
        print(f"Overall calibration: {'✓ VALID' if all_components_valid else '✗ INVALID'}")
    
        if not all_components_valid:
            
            print("\nRECOMMENDATIONS:")
            
            if not f_valid:
                print("- Recompute focal length using vanishing point orthogonality")
            if not K_valid:
                
                print("- Check principal point calculation and coordinate system")
            if not R_valid:
                
                print("- Apply SVD correction to rotation matrix")
            if not P_valid:
                
                print("- Remove incorrect P normalization, use P = K[R|t] directly")
    
        return {
                
            'valid': all_components_valid,
            'f': f,
            'K': K,
            'R': R_corrected,
            'P': P_corrected,
            'checks': {
                
                'f': f_valid,
                'K': K_checks,
                'R': R_checks, 
                'P': P_checks    
            }
        }

    # Run complete validation
    validation_results = complete_calibration_validation(f, K, R, t, P, extended_w, extended_h)

    # Use corrected matrices if validation passed
    if validation_results['valid']:
        #    
        print("\nC All matrices validated successfully!")
        K_validated = validation_results['K']
        R_validated = validation_results['R']
        P_validated = validation_results['P']
    else:
        #    
        print("\nC Validation failed - using corrected matrices")
        K_validated = validation_results['K']
        R_validated = validation_results['R']
        P_validated = validation_results['P']
    
   # visualizition function for referance person

    def visualize_reference_person(image, head_pos, foot_pos, H_real):
        
        """Visualize selected reference person with height information"""
        vis_img = image.copy()
    
    # Draw head and foot points
        cv2.circle(vis_img, (int(head_pos[0]), int(head_pos[1])), 8, (0, 255, 0), -1)  # Green for head
        cv2.circle(vis_img, (int(foot_pos[0]), int(foot_pos[1])), 8, (0, 0, 255), -1)  # Red for foot
    
        # Draw vertical line connecting head and foot
        cv2.line(vis_img, 
             (int(head_pos[0]), int(head_pos[1])), 
             (int(foot_pos[0]), int(foot_pos[1])), 
             (255, 255, 0), 2)  # Yellow line
    
        # Add labels
        cv2.putText(vis_img, 'Head', 
                (int(head_pos[0]) + 10, int(head_pos[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_img, 'Foot', 
                (int(foot_pos[0]) + 10, int(foot_pos[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
        # Add height information
        height_text = f'Height: {H_real:.2f}m'
        cv2.putText(vis_img, height_text, 
                (int(head_pos[0]) - 50, int(head_pos[1]) - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
        # Show pixel height
        pixel_height = np.sqrt((foot_pos[0] - head_pos[0])**2 + (foot_pos[1] - head_pos[1])**2)
        pixel_text = f'Pixel height: {pixel_height:.1f}px'
        cv2.putText(vis_img, pixel_text, 
                (int(head_pos[0]) - 50, int(head_pos[1]) - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the visualization
        cv2.imshow('Selected Reference Person', vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        # Save the visualization
        cv2.imwrite('reference_person.png', vis_img)
        return vis_img
    
    
    def visualize_detected_persons(image, pairs, title="Detected Persons"):
        """Visualize all detected persons with their IDs"""
        vis_img = image.copy()
    
        # Draw each person
        for idx, (hx, hy, fx, fy) in enumerate(pairs):
            # Draw head point (green)
            cv2.circle(vis_img, (int(hx), int(hy)), 8, (0, 255, 0), -1)
        
            # Draw foot point (red)
            cv2.circle(vis_img, (int(fx), int(fy)), 8, (0, 0, 255), -1)
        
            # Draw line connecting head and foot (yellow)
            cv2.line(vis_img, (int(hx), int(hy)), (int(fx), int(fy)), (0, 255, 255), 2)
        
            # Add person ID
            cv2.putText(vis_img, f'Person {idx}', 
                    (int(hx) - 30, int(hy) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
            # Add height in pixels
            pixel_height = np.sqrt((fx - hx)**2 + (fy - hy)**2)
            cv2.putText(vis_img, f'{pixel_height:.0f}px', 
                    (int(hx) + 10, int(hy) + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
        # Show the visualization
        cv2.imshow(title, vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return vis_img

    

      
    
    # --- Reference Person Selection from Separate Photo ---
    print("\n--- Reference Person Selection ---")
    ref_photo_path = input("Enter the path to the reference photo: ").strip()
    ref_selected_frame = cv2.imread(ref_photo_path)
    if ref_selected_frame is None:
        print(f"Could not read the reference image: {ref_photo_path}")
        exit()

    # Use the same process_image function as used for input photos
    processed_frame, detection_results = process_image(ref_selected_frame)
    ref_pairs = [(d['head'][0], d['head'][1], d['foot'][0], d['foot'][1]) for d in detection_results]

    #     Show detected persons
    if len(ref_pairs) == 0:
        print("No persons detected in the reference photo.")
        exit()

    # Visualize detected persons
    print("\nDetected persons in reference photo:")
    for pidx, (hx, hy, fx, fy) in enumerate(ref_pairs):
        print(f"Person {pidx}: Head=({hx},{hy}), Foot=({fx},{fy})")    

    # Display processed frame with detections
    vis_img = visualize_detected_persons(ref_selected_frame, ref_pairs, "Reference Photo - Select Person")
    cv2.imwrite('reference_persons_detected.png', vis_img)

    # Get user input for person selection
    ref_person_idx = int(input("\nEnter the person index in the reference photo: ").strip())
    if ref_person_idx < 0 or ref_person_idx >= len(ref_pairs):
        print("Invalid person index")
        exit()

    # Get real-world height
    H_real = float(input("Enter the real-world height of this person in meters (e.g., 1.70): ").strip())

    # Store reference person data
    ref_pair = ref_pairs[ref_person_idx]

    # Print selection summary
    print(f"\nReference person selected:")
    print(f"Image: {ref_photo_path}")
    print(f"Person index: {ref_person_idx}")
    print(f"2D Head position: ({ref_pair[0]}, {ref_pair[1]})")
    print(f"2D Foot position: ({ref_pair[2]}, {ref_pair[3]})")
    print(f"Real-world height (H_real): {H_real} meters")

    # Visualize selected reference person
    head_pos = (ref_pair[0], ref_pair[1])
    foot_pos = (ref_pair[2], ref_pair[3])
    ref_vis = visualize_reference_person(ref_selected_frame, head_pos, foot_pos, H_real)
    print("Reference person visualization saved as 'reference_person.png'")



####--- WE ARE TAKKING TARGET PERSON FROM ONE OF THE INPUT IMAGES ---####
# Select a foot point from input images
    print("\n--- Select a foot point from input images ---")
    print("Available input images:")
    for idx, photo_path in enumerate(photo_paths):
        # Display available images with indices
        print(f"[{idx}] {photo_path}")

# Get user input for image selection
    target_img_idx = int(input("\nEnter the image index to select foot point from: ").strip())
    if target_img_idx < 0 or target_img_idx >= len(photo_paths):
        
        #  # Validate the input index
        print("Invalid image index")
        exit()

# Show available people in selected image
    target_pairs = image_head_foot_pairs[target_img_idx]
    print(f"\nAvailable people in selected image:")
    for pidx, (hx, hy, fx, fy) in enumerate(target_pairs):
        print(f"Person {pidx}: Head point at ({hx}, {hy}), Foot point at ({fx}, {fy})")
        #print(f"Person {pidx}: Foot point at ({fx}, {fy})")
        

# Get user input for person selection
    target_person_idx = int(input("\nEnter the person index to use their foot point: ").strip())
    if target_person_idx < 0 or target_person_idx >= len(target_pairs):
        
        print("Invalid person index")
        exit()

# Get the selected foot point
    selected_foot_point = (target_pairs[target_person_idx][2], target_pairs[target_person_idx][3])
    print(f"\nSelected foot point: ({selected_foot_point[0]}, {selected_foot_point[1]})")
#Get the selected head point
    selected_head_point = (target_pairs[target_person_idx][0], target_pairs[target_person_idx][1])
    print(f"Selected head point: ({selected_head_point[0]}, {selected_head_point[1]})")
    
# store the selected head point for later use
    target_head_x, target_head_y = selected_head_point
# Store the selected foot point for later use
    target_foot_x, target_foot_y = selected_foot_point

# Visualize the selected foot point
    target_frame = cv2.imread(photo_paths[target_img_idx])
    if target_frame is not None:
        cv2.circle(target_frame, selected_foot_point, 8, (0, 0, 255), -1)
    cv2.putText(target_frame, 'Selected Foot Point', 
                (selected_foot_point[0] + 10, selected_foot_point[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Selected Foot Point', target_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    # Quick fix: Verify we're using calibration from the correct image
    if target_img_idx != 1:  # Since your calibration was computed for image 1
        
        print(f"ERROR: Calibration was computed for image 1, but selecting from image {target_img_idx}")
        print("Please select foot point from image 1 or recompute calibration")
        exit()

    print(f"Confirmed: Using calibration and foot point both from image {target_img_idx}")
    
    



        
    def compute_f3D_ray_intersection(K, R, t, foot_2d_extended, debug=True):
        """
        Alternative f₃D computation using ray-plane intersection
        This bypasses the P^T P inversion entirely
        """
        if debug:
            print(f"\n--- RAY-PLANE INTERSECTION METHOD ---")
    
        # Convert to homogeneous coordinates
        f2D = np.array([foot_2d_extended[0], foot_2d_extended[1], 1.0])
    
        try:
            # Step 1: Back-project pixel to camera coordinate ray
            K_inv = np.linalg.inv(K)
            ray_camera = K_inv @ f2D
            ray_camera = ray_camera / np.linalg.norm(ray_camera)  # Normalize
        
            if debug:
                print(f"Ray in camera coordinates: {ray_camera}")
        
            # Step 2: Transform ray to world coordinates
            ray_world = R.T @ ray_camera  # R.T because we go from camera to world
        
            if debug:
                print(f"Ray in world coordinates: {ray_world}")
        
            # Step 3: Camera center in world coordinates
            camera_center = -R.T @ t.flatten()
        
            if debug:
                print(f"Camera center in world: {camera_center}")
        
            # Step 4: Intersect ray with ground plane (Z = 0)
            if abs(ray_world[2]) < 1e-8:
                print("Error: Ray parallel to ground plane")
                return None, False
        
            # Parametric ray: point = camera_center + λ * ray_world
            # For ground plane Z = 0: camera_center[2] + λ * ray_world[2] = 0
            lambda_param = -camera_center[2] / ray_world[2]
        
            if debug:
                print(f"Ray parameter λ: {lambda_param}")
        
            # Compute 3D intersection point
            f3D = camera_center + lambda_param * ray_world
            f3D[2] = 0.0  # Ensure exactly on ground plane
        
            if debug:
                print(f"f₃D result: ({f3D[0]:.3f}, {f3D[1]:.3f}, {f3D[2]:.3f})")
        
            return f3D, True
        
        except Exception as e:
            print(f"Ray intersection method failed: {e}")
            return None, False
        
        
        # FUNCTION FOR VISUALIZE THE PROJECTED F3D TO 2D:
    def visualize_reprojection(image, original_point, reprojected_point, margin_w, title="Reprojection Visualization"):
        
        """Visualize original and reprojected points"""
        h, w = image.shape[:2]
        # Create extended canvas
        vis_img = np.zeros((h, w + 2 * margin_w, 3), dtype=np.uint8)
        vis_img[:, margin_w:margin_w+w] = image  # Place original image in center
    
        
        
        # Draw original point (red), shifted into extended canvas
        orig_x_ext = int(round(original_point[0] + margin_w))
        orig_y_ext = int(round(original_point[1]))
        cv2.circle(vis_img, (orig_x_ext, orig_y_ext), 8, (0, 0, 255), -1)
        cv2.putText(vis_img, 'Original', (orig_x_ext + 10, orig_y_ext), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        

            
        # Draw reprojected point (green), already in extended canvas coordinates
        reproj_x_ext = int(round(reprojected_point[0]))
        reproj_y_ext = int(round(reprojected_point[1]))
        cv2.circle(vis_img, (reproj_x_ext, reproj_y_ext), 8, (0, 255, 0), -1)
        cv2.putText(vis_img, 'Reprojected', (reproj_x_ext + 10, reproj_y_ext), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw line between points
        cv2.line(vis_img, (orig_x_ext, orig_y_ext), (reproj_x_ext, reproj_y_ext), (255, 255, 0), 2)

    

        reproj_original = np.array([reprojected_point[0] - margin_w, reprojected_point[1]])
        error = np.linalg.norm(reproj_original - np.array(original_point))
        cv2.putText(vis_img, f'Error: {error:.1f}px', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(title, vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return vis_img

        
    x, y = selected_foot_point 
    #f2D = np.array([x, y, 1.0])  # Foot point in homogeneous coordinates
    x_extended = x + margin_w  # Add horizontal margin offset
    f2D = np.array([x_extended, y, 1.0])  # Use extended x-coordinate
        
    print(f"\n--- COMPUTING F3d ---")
    print(f"Original foot point: ({x}, {y})")
    print(f"Extended foot point: ({x_extended}, {y})")
    print(f"Principal point used: ({px}, {py})")
        
    # Ensure P is properly formed and normalized
    if P is None or np.any(np.isnan(P)) or np.any(np.isinf(P)):
        raise ValueError("Invalid projection matrix P")


    # Try the ray intersection method:
    f3D_ray, success_ray = compute_f3D_ray_intersection(K, R, t, f2D)

    if success_ray:
        print("✓ Ray intersection method successful!")
        # Validate by projecting back
        f3D_hom = np.array([f3D_ray[0], f3D_ray[1], f3D_ray[2], 1.0])
        reprojected_hom = P @ f3D_hom
        if abs(reprojected_hom[2]) > 1e-8:
            reprojected_2d = reprojected_hom[:2] / reprojected_hom[2]
            reprojected_original = (reprojected_2d[0] - margin_w, reprojected_2d[1])
            error = np.linalg.norm(np.array(reprojected_original) - np.array(selected_foot_point))
            
            print(f"Reprojected foot point: {reprojected_2d}")
            print(f"Original foot point: {selected_foot_point}")
            print(f"reprojected original foot point: {reprojected_original}")
            
            print(f"Ray method reprojection error: {error:.2f} pixels")
            
            
            vis_img = visualize_reprojection(
            target_frame, 
            selected_foot_point,
            (reprojected_2d[0], reprojected_2d[1]),
            margin_w,
            "f3D Reprojection Check"
        )
        # Save visualization
        cv2.imwrite('f3D_reprojection.png', vis_img)
            

    


    #Computing reference 3D person's foot point:

    def compute_ref_3D_footpoint(K, R, t, ref_foot_2d, debug=True):
        """
        Compute 3D reference foot point using ray-plane intersection
        """
        if debug:
            print("\n--- Computing Reference 3D Point Using Ray Intersection ---")
    
        # Convert to homogeneous coordinates
        f2D = np.array([ref_foot_2d[0], ref_foot_2d[1], 1.0])
    
        try:
            # 1. Back-project pixel to ray in camera coordinates
            K_inv = np.linalg.inv(K)
            ray_camera = K_inv @ f2D
            ray_camera = ray_camera / np.linalg.norm(ray_camera)
        
            if debug:
                print(f"Ray direction (camera space): {ray_camera}")
        
            # 2. Transform ray to world coordinates
            ray_world = R.T @ ray_camera
        
            # 3. Get camera center in world coordinates
            C = -R.T @ t.flatten()
        
            if debug:
                
                print(f"Camera center (world): {C}")
                print(f"Ray direction (world): {ray_world}")
        
            # 4. Intersect with ground plane (Z = 0)
            # Using parametric equation: C + λ * ray_world = [X, Y, 0]
            # Solve for λ when Z component = 0
            lambda_param = -C[2] / ray_world[2]
        
            # 5. Compute intersection point
            ref_3D = C + lambda_param * ray_world
            ref_3D[2] = 0.0  # Ensure exactly on ground plane
        
            if debug:
                print(f"Reference 3D point: ({ref_3D[0]:.3f}, {ref_3D[1]:.3f}, {ref_3D[2]:.3f})")
            
            return ref_3D, True
        
        except Exception as e:
            print(f"Error computing reference 3D point: {e}")
            return None, False

    # Get reference foot point in extended coordinates
    ref_x, ref_y = ref_pair[2], ref_pair[3]  # Using foot coordinates from ref_pair
    ref_x_extended = ref_x + margin_w  # Add margin offset
    ref_foot_2d = (ref_x_extended, ref_y)
    print(f"\n--- CHECKING REFERANCE FOOT POINTS BEFORE REF f3d ---")
    print(f"Reference foot point (original): ({ref_x}, {ref_y})")
    print(f"Reference foot point (extended): ({ref_x_extended}, {ref_y})")
  
    ref_3D, success = compute_ref_3D_footpoint(K, R, t, ref_foot_2d)
    if success:
        print("\nValidating reference 3D point:")
        # Project back to 2D to verify
        ref_3D_hom = np.append(ref_3D, 1.0)
        projected = P @ ref_3D_hom
        projected_2d = projected[:2] / projected[2]
        # Convert back to original coordinates
        projected_original = (projected_2d[0] - margin_w, projected_2d[1])
    
        error = np.linalg.norm(np.array(projected_original) - np.array([ref_x, ref_y]))
        print(f"Reprojection error: {error:.2f} pixels")
    
        if error < 5.0:
            
            print("✓ Reference 3D point validation passed")
            print(f"reprojected foot point: {projected_2d}")
            print(f"projected foot point: {projected_original}")
            
            # Visualize ref3D reprojection
            vis_img = visualize_reprojection(
            ref_selected_frame,
            (ref_x, ref_y),    
            (projected_2d[0], projected_2d[1]),
            margin_w,"Reference Point Reprojection Check")
            # Save visualization
            cv2.imwrite('ref3D_reprojection.png', vis_img)   
        else:
            print("⚠ Large reprojection error")
    else:
        print("Failed to compute reference 3D point")
        
        
    
        
    
    print("\n--- computing ref_3D Head Point Computation ---")
    # Suppose:
    # f3D_ref : numpy array shape (3,), known foot position
    # H_ref : float, known real height
    
    h3D_ref = ref_3D - np.array([0, 0, H_real])
    print(f"Computed 3D head point: ({h3D_ref[0]:.3f}, {h3D_ref[1]:.3f}, {h3D_ref[2]:.3f})")

    # Check vertical displacement (should equal H_ref)
    vertical_diff = h3D_ref[2] - ref_3D[2]
    print(f"Vertical displacement (z): {vertical_diff:.3f} meters")

    # Assert the foot point is near ground (z ≈ 0)
    if abs(ref_3D[2]) > 1e-3:
        print("Warning: foot point z-coordinate not near zero, check ground plane assumption.")

    # Assert head point vertical correctness
    if not np.isclose(vertical_diff, H_real, atol=1e-3):
        print("Warning: head point vertical displacement does not match input height H_ref.")
    else:
        print("3D head point computation valid.")

    print("\n--- Projecting ref_3D Head Point to 2D ---")
    # P: 3x4 projection matrix
    # h3D_ref: numpy array shape (3,)

    h3D_hom = np.append(h3D_ref, 1.0)  # make homogeneous
    projected = P @ h3D_hom           # shape (3,)
    
    # Normalize to get pixel coordinates
    h2D_ref = projected[:2] / projected[2]

    print(f"Projected 2D ref_head point: ({h2D_ref[0]:.1f}, {h2D_ref[1]:.1f})")

    # Check if within image bounds
    img_h, img_w = ref_selected_frame.shape[:2]  # your image dimensions
    print(f"Image dimensions: {img_w}x{img_h}")
    
    #If working with extended canvas (use these dimensions for bounds checking/visualization):
    #img_h, img_w = ref_selected_frame.shape[:2]        # original image height and width
    img_w_ext = img_w + 2 * margin_w                   # extended width
    img_h_ext = img_h                                  # extended height (usually unchanged)
    print(f"Projected 2D ref_head point (extended canvas): ({h2D_ref[0]:.1f}, {h2D_ref[1]:.1f})")

    within_bounds = (0 <= h2D_ref[0] < img_w_ext) and (0 <= h2D_ref[1] < img_h_ext)
    if within_bounds:
        print("Projected 2D ref_head point lies within image bounds.")
    else:
        print("Warning: projected head point outside image bounds.")

    
    
    img_vis = np.zeros((img_h_ext, img_w_ext, 3), dtype=np.uint8)  # blank extended canvas
    # If you want to show the original image in the center of the extended canvas:
    img_vis[:, margin_w:margin_w+img_w] = ref_selected_frame
    cv2.circle(img_vis, (int(round(h2D_ref[0])), int(round(h2D_ref[1]))), 5, (0,255,0), -1)
    cv2.imshow("Projected Head Point (Extended Canvas)", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    
    
    print(f"\n--- Height Estimation ---")
    
    y_target_foot = selected_foot_point[1]
    y_target_head = selected_head_point[1]
    
    y_ref_head_proj = h2D_ref[1]
    
    numerator = abs(y_ref_head_proj - y_target_foot)
    denominator = abs(y_target_head - y_target_foot)
    Ho = (numerator / denominator) * H_real
    
    

    print(f"Estimated Height H_o: {Ho:.3f} meters")
        
    
