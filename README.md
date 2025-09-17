Smarter Safety: Real-Time PPE Detection System
📌 Overview

This project develops an AI-powered Personal Protective Equipment (PPE) monitoring system utilizing YOLOv8 and Streamlit. The system helps construction and industrial sites automatically detect whether workers are wearing required safety gear (hard hats, masks, vests) in real time, reducing risks and improving compliance.

🚧 Why This Matters

Manual PPE checks are slow, inconsistent, and prone to human error.

Supervisors can’t monitor all workers simultaneously.

Delayed detection of non-compliance can lead to serious accidents.

Our solution uses computer vision to provide real-time, automated PPE compliance monitoring, ensuring safer worksites.

⚙️ Features

✅ YOLOv8 Detection Model – Detects PPE items and missing gear.
✅ Compliance Categories –

🟢 Compliant (all PPE present)

🟡 Partially compliant (1–2 items missing)

🔴 Non-compliant (no PPE)
✅ Streamlit Dashboard – Upload images, run detections, and view results.
✅ Visual Feedback – Color-coded bounding boxes, compliance counts, graphs, and alerts.
✅ Performance Metrics – Precision, recall, IoU, F1-score, confusion matrix.
✅ Continuous Improvement Plan – Handles data drift, rule updates, and new site conditions.

📂 Project Workflow

Problem Understanding – Interviews & field observations.

Data Preparation – Cleaning, resizing (224x224), augmentation, normalization.

Model Training – YOLOv8 with optimizations (early stopping, label smoothing).

Evaluation – Precision-recall, IoU, confusion matrix, heatmaps.

Dashboard Development – Streamlit-based interface for supervisors.

Deployment – Real-world testing with ongoing updates.

📊 Evaluation Metrics

Precision – Minimizing false alarms.

Recall – Detecting true violations.

F1-Score – Balancing precision and recall.

IoU – Measuring detection accuracy.

🖥️ Streamlit Dashboard Preview

Upload PPE images.

View detections in real-time with compliance highlights.

Monitor compliance trends via graphs.

Receive urgent alerts for non-compliance.

🚀 Deployment & Maintenance

Deployable on construction/industrial sites with cameras.

Periodic retraining with updated datasets.

Adjustable thresholds for evolving PPE standards.

User feedback integrated for improvement.

🛠️ Tech Stack

Python

YOLOv8 (Ultralytics)

Streamlit

OpenCV

NumPy, Pandas, Matplotlib

📘 Storyboard (At a Glance)

Problem: Manual checks miss violations.

Vision: AI-powered real-time detection.

Field Insights: Supervisor limitations.

Data Prep: Cleaning & augmentation.

Training: YOLOv8 compliance model.

Evaluation: Metrics & visualization.

Interface: Streamlit dashboard.

Deployment: Alerts & continuous updates.

Future: Smarter, safer worksites.

👥 Contributors

Team Members: Year-2 AI/ML Project Group

Role: Data preparation, model training, dashboard development, deployment, and testing.

📄 License

This project is licensed under the MIT License – feel free to use, modify, and share with attribution.
