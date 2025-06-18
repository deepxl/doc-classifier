# Architecture Overview

This document provides a brief overview of the technical architecture for the Gemini Document Classifier.

## Core Philosophy

The project is designed around three core principles:

1.  **Speed**: Every decision is optimized for the lowest possible latency.
2.  **Simplicity**: The architecture is intentionally kept simple and direct to avoid unnecessary complexity and overhead.
3.  **Accuracy**: While speed is critical, the system must maintain the highest possible level of classification accuracy.

## Key Components

The system is composed of a few key parts that work together.

### 1. Classification Engine (`src/core/document_classifier.py`)

This is the heart of the system. The `UltraFastDocumentClassifier` class encapsulates all the logic for making classification requests to the Google Gemini API.

- It handles image loading and preparation.
- It constructs the API request using the appropriate model, prompt, and parameters from the configuration.
- It leverages Gemini's **structured output** feature to receive a predictable JSON response, which eliminates the need for fragile text parsing and significantly improves speed and reliability.
- It includes methods for classifying single images and batch processing multiple images in parallel using a `ThreadPoolExecutor`.

### 2. Configuration System (`src/config/`)

Configuration is managed in a modular way, allowing for easy updates and experiments.

- **`models.py`**: This file is the source of truth for all supported Gemini models and their associated parameters. It defines different parameter sets (e.g., `optimal`, `standard`) for each model, which have been tuned through extensive testing.
- **`prompts.py`**: Contains the prompt templates. The `detailed` prompt is the production default, as it was found to be the fastest and most reliable during optimization tests.
- **`categories.py`**: Defines the list of possible document types. This provides a single source of truth for the classification schema.
- **`structured_output.py`**: Defines the JSON schemas that are sent to the Gemini API to enforce the structure of the response. This is a critical component for both speed and reliability.

### 3. Testing Script (`scripts/test-models.py`)

This is the primary tool for evaluating and validating the performance of the classifier.

- It tests a hardcoded set of high-performance Gemini models.
- It compares the classifier's output against a `ground-truth.json` file.
- It calculates and reports on key machine learning metrics (Accuracy, Precision, Recall, F1-score) as well as performance metrics (latency, throughput).
- It serves as a quality gate to ensure that changes to the configuration or codebase do not negatively impact performance.

## Data Flow

The classification process follows a simple data flow:

1.  An image path is provided to the `UltraFastDocumentClassifier`.
2.  The classifier reads the image and selects the configured model, prompt, and parameters.
3.  It constructs a request to the Gemini API, including the image data and the desired JSON output schema.
4.  The Gemini API processes the request and returns a JSON object containing the `document_type` and a `confidence` score.
5.  The classifier parses the response into a simple data object and returns it.
