# Real-Time-Helmet-Violation-Detection-and-Automated-Number-Plate-Recognition
An automated, edge-tier intelligent transportation system (ITS) framework designed for real-time traffic law enforcement. This project implements a parallel dual-node deep learning pipeline that concurrently detects helmet compliance violations and isolates vehicle license plates from live video streams. Identified infractions are processed using an optimized OCR pipeline and immediately sent to a cloud notification gateway for automated civic penalty enforcement.
🛠️ System Architecture & Core Modules

The system architecture is decoupled into four highly optimized, interconnected modules to maximize frame throughput and eliminate processing delays:
1. Dual-Node Deep Learning Detection Module

    Helmet Compliance Node: Deploys a customized YOLOv8-Small model to classify riders (With Helmet vs. Without Helmet) under complex, unconstrained outdoor environments with an accuracy of 89.60%.

    License Plate Localization Node: Implements an ultra-lightweight YOLOv8-Nano model specifically trained for high-precision bounding box regression over vehicle registration plates, achieving a precision (mAP50​) of 99.50%.

    Normalized Data Labeling: Annotates visual targets using a standard 5-element vector format: [class_id, x_center, y_center, width, height].

2. Performance Optimization & Tracking Module

    Asynchronous Multi-Threading: Isolates heavy video capture and tracking routines to a high-priority main thread while offloading heavy OCR processing tasks to a non-blocking background worker queue. This eliminates "loop starvation" and preserves a real-time production velocity of ≥30 FPS.

    Temporal Identity Tracking Buffer: Features a 45-frame occlusion decay counter to maintain unique vehicle tracking IDs across sequential frames, preventing duplicate penalty logging if a target is briefly blocked by another vehicle.

    Granular Model Evaluation: Monitored via highly specialized metrics focusing heavily on Sensitivity (Recall) to minimize missed violations and Specificity to eliminate false citations.

3. Image Preprocessing & OCR Extraction Module

    Spatial Enhancement Layers: Applies a 15% canvas padding expansion around the license plate bounding box to protect edge text from cropping errors, followed by Cubic Interpolation upscaling.

    Adaptive Thresholding: Leverages Otsu's Binarization to strip out real-world video noise, heavy shadow castings, and bright lens glares by translating crops into clean high-contrast black-and-white pixels.

    Deterministic Syntax Correction Engine: Passes raw characters through a regular expression (Regex) parser mapped to regional RTO layout standards to automatically fix lookalike OCR character confusion (e.g., swapping '0' for 'O' or '1' for 'I'), elevating final text recognition accuracy to 95.2%.

4. Backend Logging & Cloud Notification Module

    Relational Schema Mapping: Persists unique infraction profiles, accurate timestamps, localized crops, and confidence scores into a structured local relational database.

    Cloud Gateway Communication: Executes secure, asynchronous HTTPS REST API requests to the Twilio API gateway to dispatch automated cellular SMS text notifications to the registered offender within a strict end-to-end timeline of ≤1.8 seconds from initial camera localization.

📊 Production Performance Metrics
Pipeline Component	Metric Tested	Final Value Obtained
YOLOv8-Nano (License Plate Node)	mAP50​	99.50%
YOLOv8-Small (Helmet Node)	Classification Accuracy	89.60%
EasyOCR + Regex Parser	Absolute Text Accuracy	95.2%
End-to-End Latency	Detection to SMS Dispatch	≤1.8 seconds
System Throughput	Video Processing Velocity	≥30 FPS
💻 Technical Stack

    Frameworks & Core AI: PyTorch, Ultralytics (YOLOv8 Architecture), EasyOCR Engine, OpenCV

    Languages & Mechanics: Python, Multithreading Protocols (Queue / Threading), Regular Expressions (Regex)

    Data & Infrastructure: Twilio API Gateway, Cloud-integrated HTTPS REST Protocols, Relational SQL Datastores

🚀 Getting Started
Prerequisites

    Python 3.8+

    CUDA-supported NVIDIA GPU (highly recommended for dual-node real-time frame rates)

Installation & Deployment

    Clone the Repository:
    Bash

    git clone https://github.com/yourusername/helmet-anpr-surveillance.git
    cd helmet-anpr-surveillance

    Install Core Requirements:
    Bash

    pip install -r requirements.txt

    Configure Gateway Secrets:
    Create a secure .env configuration file in your project's root directory:
    Code snippet

    TWILIO_ACCOUNT_SID=your_twilio_sid_here
    TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
    TWILIO_PHONE_NUMBER=your_twilio_phone_number_here
    TARGET_PHONE_NUMBER=receiver_phone_number_here

    Launch the Real-Time Pipeline:
    Bash

    python main.py --source traffic_feed.mp4

🔍 Engineering Evaluation Insight

During initial training phases, an artificial 100% accuracy score indicating a severe data leakage anomaly was detected. By isolating features, cleaning out dependent evaluation variables, and shifting structural optimization milestones directly toward distinct sensitivity and specificity margins, the final dual-node configuration has been hardened to securely generalize over entirely unseen, unconstrained real-world traffic flows without any overfitting.

⚠️ System Limitations & Future Scope

While the system achieves high accuracy and real-time throughput, the following edge-case limitations have been identified under unconstrained real-world deployment scenarios:

  + Extreme Weather Decoupling: Heavy downpours, thick fog, or dense dust storms can significantly scatter light, leading to a degradation in bounding box regression accuracy for the YOLOv8 nodes and introducing character noise in the OCR engine.

  + Severe Occlusion & Crowding: If a violator is closely tailgating a large commercial vehicle (e.g., a bus or truck) where the license plate is completely blocked from the camera's line of sight for longer than 45 consecutive frames, the tracking link breaks and logging fails.

  + Angled/Perspective Distortion: Extreme camera mounting angles (≥45∘ relative to the vehicle's trajectory) create structural perspective distortion. While cubic interpolation mitigates this, highly skewed angles can cause character misinterpretation during OCR decoding.

  + Hardware Dependency: Maintaining a production-grade throughput of ≥30 FPS across parallel dual-node models relies heavily on CUDA-accelerated GPU resources; running the pipeline exclusively on standard edge CPU hardware degrades frame velocity.

  + Non-Standard Plate Formats: The syntax correction engine is deterministically optimized for regional RTO alphanumeric layouts. Outdated, customized, handwritten, or severely broken license plates may fail layout parsing checks.
