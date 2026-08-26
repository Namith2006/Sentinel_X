# Sentinel X - AI Security Dashboard 🛡️

## Project Overview
Sentinel X is a hybrid AI security application designed to detect malicious phishing URLs and identify AI-generated deepfake images. It features a modern, asynchronous web dashboard powered by a FastAPI backend.

### Core Engines:
*   **Phishing Detection:** A Scikit-Learn machine learning model combined with a rule-based whitelist for zero-latency domain verification.
*   **Deepfake Vision:** Integrates with Hugging Face's advanced vision models via the Inference API to analyze facial artifacts and classify images as real or AI-generated.

## Tech Stack
*   **Backend:** Python, FastAPI, Uvicorn
*   **Machine Learning:** Scikit-Learn, Hugging Face API
*   **Frontend:** HTML5, CSS3 (Dark Mode UI), Vanilla JavaScript (Fetch API)

## How to Run the Project locally

**1. Install Dependencies**
Ensure you have Python installed, then run the following command to install all required libraries: