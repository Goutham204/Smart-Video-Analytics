# Smart Video Analytics System
A serverless computer vision system for real-time people detection, crowd monitoring, and bidirectional line-crossing analytics — built on AWS Lambda, S3, and DynamoDB.

**🔗 Live demo:** https://d1x4qk682zvr1a.cloudfront.net

## What it does

- Detects and counts people in real time from a live webcam feed or an uploaded video file, using YOLOv8
- Tracks individual people across frames with a custom centroid-tracking algorithm, enabling accurate directional counting (IN / OUT) across a configurable line — without double-counting the same person across frames
- Supports three preset scenarios (mall entrance, office door, metro gate), each with its own default crowd-alert threshold
- Let's a user toggle detection, crowd alerts, and line-crossing independently at request time, and adjust the crowd threshold live
- Persists annotated detection frames to S3 and structured analytics (counts, timestamps, alert status) to DynamoDB

