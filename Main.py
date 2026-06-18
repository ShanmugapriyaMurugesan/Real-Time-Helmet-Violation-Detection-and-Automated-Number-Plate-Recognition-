import cv2
import math
import cvzone
import numpy as np
import easyocr
import re
from collections import Counter
from ultralytics import YOLO
from twilio.rest import Client  # Added Twilio Import

# 1. MODEL INITIALIZATION (Forced to CPU for Environment Stability)
helmet_model = YOLO("/Motorcycle Helmet and License plate detection.v6-datasetv6.yolov8/helmetdetection/Weights/best.pt")
plate_model = YOLO("License Plate Recognition.v13i.yolov8/helmet_model_final/liceplate/Weights/best.pt")


# TWILIO CONFIGURATION (Replace placeholder tokens with live details)

TWILIO_ACCOUNT_SID = "AC7xxxxxxxxxxxxxxxx"
TWILIO_AUTH_TOKEN = "e7exxxxxxxxxxxxx"
TWILIO_NUMBER = "+1 xxxx xxxxx"
MY_PHONE_NUMBER = "xxxxxxxxxx"  

try:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
except Exception as e:
    print(f"Twilio failed to initialize: {e}. Running without live SMS alerts.")
    twilio_client = None

# Tracks which unique Track IDs have already sent out an SMS alert
sent_sms_alerts = set()


# Force CPU execution to prevent hidden backend crashes if PyTorch CUDA links are broken
reader = easyocr.Reader(['en'], gpu=False)
helmet_classes = ['With Helmet', 'Without Helmet']

# Hardcoded Spatial Substitution Dictionaries
to_char = {'1': 'I', '4': 'A', '6': 'G', '8': 'B', '0': 'O', '5': 'S', '2': 'Z'}
to_num  = {'L': '4', 'J': '0', 'I': '1', 'Z': '2', 'O': '0', 'S': '5', 'B': '8', 'G': '6', 'D': '0'}

# 2. FILE INTERFACE PROPERTIES
video_path = "D:/M.Sc/Major_project/project/Motorcycle Helmet and License plate detection.v6-datasetv6.yolov8/Media/13044846_1280_720_30fps.mp4"
output_path = "D:/M.Sc/Major_project/project/Motorcycle Helmet and License plate detection.v6-datasetv6.yolov8/output_final_combined1.mp4"

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 3. SOFTWARE-BASED STABILIZATION MEMORY BANKS
plate_memory = {}        # Holds all historic text variants per ID
track_lost_counter = {}  # Manually counts how long an ID has been missing

active_lock_on_id = None
active_ocr_text = "Scanning..."
active_h_status = "Unknown"
active_plate_crop = None
frame_count = 0

def get_center(box):
    return int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)

def associate_helmet(plate_box, helmet_list):
    pcx, pcy = get_center(plate_box)
    best_status = "Unknown"
    min_dist = 300  # Expanded spatial radius to accommodate fast-moving vehicles
    for h_box in helmet_list:
        hx1, hy1, hx2, hy2, hlabel = h_box
        hcx, hcy = get_center((hx1, hy1, hx2, hy2))
        dist = math.sqrt((pcx - hcx)**2 + (pcy - hcy)**2)
        if dist < min_dist:
            min_dist = dist
            best_status = hlabel
    return best_status

# Added function to handle SMS alerts
def send_sms_alert(track_id, license_plate_text):
    """Sends an automated SMS notification for helmet violations"""
    if not twilio_client:
        return
    try:
        message_body = (
            f"🚨 TRAFFIC VIOLATION DETECTED 🚨\n"
        )
        message = twilio_client.messages.create(
            body=message_body,
            from_=TWILIO_NUMBER,
            to=MY_PHONE_NUMBER
        )
        print(f"📱 Violation Alert SMS sent successfully to {MY_PHONE_NUMBER}! SID: {message.sid}")
    except Exception as e:
        print(f"⚠️ Failed to send SMS for Track ID {track_id}: {e}")

# 4. EXECUTION STREAM LOOP
while True:
    success, frame = cap.read()
    if not success: 
        break

    frame_count += 1
    output_frame = frame.copy()

    # STEP A: Run Helmet Detections 
    h_results = helmet_model(frame, verbose=False)
    current_helmets = []
    for r in h_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = helmet_classes[int(box.cls[0])]
            current_helmets.append((x1, y1, x2, y2, label))
            
            h_color = (0, 255, 0) if label == 'With Helmet' else (0, 0, 255)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), h_color, 2)
            cv2.putText(output_frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, h_color, 2)

    # STEP B: Run Plate Detections (Using only valid base YOLO arguments)
    p_results = plate_model.track(frame, persist=True, conf=0.20, verbose=False)
    
    seen_ids_in_this_frame = set()

    if p_results and p_results[0].boxes.id is not None:
        p_boxes = p_results[0].boxes.xyxy.cpu().numpy().astype(int)
        p_ids = p_results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(p_boxes, p_ids):
            seen_ids_in_this_frame.add(track_id)
            track_lost_counter[track_id] = 0 # Reset frame-loss timer
            
            px1, py1, px2, py2 = box
            
            # SOLUTION 1: Add a 15% pixel padding cushion around the box to stop text clipping
            pad_w = int((px2 - px1) * 0.15)
            pad_h = int((py2 - py1) * 0.15)
            
            plate_crop = frame[max(0, py1-pad_h):min(frame_height, py2+pad_h), 
                               max(0, px1-pad_w):min(frame_width, px2+pad_w)]
            
            if plate_crop.size > 0:
                # SOLUTION 2: Digital Image Restoration (Upscale + Otsu Binarization)
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                cleaned_image = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                ocr_res = reader.readtext(cleaned_image)
                
                for entry in ocr_res:
                    confidence = entry[2]
                    if confidence < 0.30:  # Drop noisy, low-confidence frames
                        continue
                        
                    raw_text = entry[1].upper().replace(" ", "").strip()
                    raw_text = "".join(c for c in raw_text if c.isalnum())
                    
                    # Apply length filter optimized for Indian state formatting constraints
                    if 9 <= len(raw_text) <= 10:
                        # SOLUTION 3: Positional Syntax Alignment for Indian Plates
                        text_list = list(raw_text)
                        for i in range(len(text_list)):
                            if i < 2 and text_list[i] in to_char:     # Positions 0-1 must be letters
                                text_list[i] = to_char[text_list[i]]
                            elif 2 <= i < 4 and text_list[i] in to_num: # Positions 2-3 must be numbers
                                text_list[i] = to_num[text_list[i]]
                        
                        cleaned_text = "".join(text_list)
                        
                        if track_id not in plate_memory:
                            plate_memory[track_id] = []
                        plate_memory[track_id].append(cleaned_text)
                        
                        # SOLUTION 4: Statistical Majority Voting Filter
                        most_stable_text = Counter(plate_memory[track_id]).most_common(1)[0][0]
                        
                        active_lock_on_id = track_id
                        active_ocr_text = most_stable_text
                        active_h_status = associate_helmet(box, current_helmets)
                        active_plate_crop = cv2.resize(plate_crop, (150, 60))

                        # --- LIVE SMS TRIGGER LOGIC ---
                        # Verify the rider has been mapped to 'Without Helmet' and hasn't been messaged yet
                        if active_h_status == "Without Helmet" and track_id not in sent_sms_alerts:
                            send_sms_alert(track_id, active_ocr_text)
                            sent_sms_alerts.add(track_id)

        # Render Tracking Enclosures
        for box, track_id in zip(p_boxes, p_ids):
            px1, py1, px2, py2 = box
            if track_id == active_lock_on_id:
                cvzone.cornerRect(output_frame, (px1, py1, px2-px1, py2-py1), l=15, t=3, rt=2, colorR=(0, 0, 255))
                cv2.putText(output_frame, "LOCK ON", (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.rectangle(output_frame, (px1, py1), (px2, py2), (0, 255, 0), 1)

    # STEP C: SOFTWARE OCULUSION MANAGEMENT (Manual replacement for max_age)
    all_tracked_ids = list(track_lost_counter.keys())
    for t_id in all_tracked_ids:
        if t_id not in seen_ids_in_this_frame:
            track_lost_counter[t_id] += 1
            if track_lost_counter[t_id] > 45: # Hold tracking history for 45 frames of occlusion
                track_lost_counter.pop(t_id, None)
                plate_memory.pop(t_id, None)
                if active_lock_on_id == t_id:
                    active_lock_on_id = None

    # STEP D: GENERATE INTERFACE SURFACE PANEL (Top Right Box)
    if active_plate_crop is not None and active_lock_on_id is not None:
        overlay_h, overlay_w = 180, 340
        pip = np.zeros((overlay_h, overlay_w, 3), dtype=np.uint8)
        pip[:] = (35, 35, 35) # Dark gray background matrix
        
        # Overlay the cropped license plate image segment
        pip[15:75, 15:165] = active_plate_crop
        
        # Render the extracted text data directly inside the dashboard panel
        cv2.putText(pip, f"NUMBER: {active_ocr_text}", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        # Compute anchoring coordinates for the top-right position dynamically
        oy, ox = 30, frame_width - overlay_w - 30
        output_frame[oy:oy+overlay_h, ox:ox+overlay_w] = pip
        cv2.rectangle(output_frame, (ox, oy), (ox+overlay_w, oy+overlay_h), (0, 0, 255), 2)

    # STEP E: SAVE OUTPUT STREAMS
    out.write(output_frame)
    cv2.imshow("Helmet Detection and Number Plate Recognition Pipeline", output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
out.release()
cv2.destroyAllWindows()