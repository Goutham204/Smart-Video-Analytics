# Smart Video Analytics System

A serverless, AWS-deployed computer vision system for real-time people counting, crowd monitoring, and line-crossing analytics — rebuilt from a Streamlit prototype into a fully cloud-native production system.

🔗 **Live Demo:** https://d1oxoi6wvob06g.cloudfront.net

---

## Overview

This project started as a local Streamlit application with real limitations — no live webcam support in the cloud, a broken line-crossing counter that miscounted on every frame, no persistent storage, and no path to real deployment.

I rebuilt it from the ground up as a properly architected, cloud-native system using AWS serverless infrastructure — keeping costs at effectively $0 when idle while supporting real-time inference via a live webcam or uploaded video.

---

## What it does

- Detects and counts people in real time from a live webcam feed or an uploaded video file, using YOLOv8
- Tracks individual people across frames with a custom centroid-tracking algorithm, enabling accurate directional counting (IN / OUT) across a configurable line — without double-counting the same person across frames
- Supports three preset scenarios (mall entrance, office door, metro gate), each with its own default crowd-alert threshold
- Let's a user toggle detection, crowd alerts, and line-crossing independently at request time, and adjust the crowd threshold live
- Persists annotated detection frames to S3 and structured analytics (counts, timestamps, alert status) to DynamoDB

---

## AWS Architecture

```
Browser (CloudFront HTTPS)
        ↓
Static Frontend (S3 + CloudFront)
        ↓
Lambda Function URL
        ↓
AWS Lambda — Docker Container
(FastAPI + YOLOv8 + OpenCV + PyTorch)
        ↓
S3 (annotated images) + DynamoDB (analytics logs)
```

## Tech Stack

* **Programming Language:** Python 3.10
* **Object Detection:** YOLOv8 (Ultralytics)
* **Computer Vision / ML:** PyTorch, YOLOv8, OpenCV
* **API Framework:** FastAPI
* **Containerization:** Docker
* **Cloud Compute:** AWS Lambda
* **Storage:** AWS S3, Amazon DynamoDB
* **CDN & HTTPS:** AWS CloudFront
* **Data Processing:** Pandas, NumPy

---

## Use Cases

- **Retail & malls** — footfall counting, crowd density monitoring at entrances
- **Offices & buildings** — door-level occupancy tracking, access monitoring
- **Metro & transit** — gate-level passenger flow analytics
- **Security & surveillance** — real-time crowd alert triggers

---

## License

This project is open-source under the **MIT License**.
