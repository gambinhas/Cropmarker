# Supervisor-in-the-Loop Annotation Aggregator

This project provides a comprehensive tool for aggregating, validating, and analyzing crowdsourced geospatial image annotations. It features a powerful graphical user interface (GUI) for expert supervisors to review annotations and a probabilistic backend using the Dawid-Skene (DS) model to intelligently fuse data from multiple raters with varying expertise and consistency.

## Features

*   **Probabilistic Data Fusion:** Implements the Dawid-Skene algorithm to move beyond simple voting and calculate the most probable "true" label for each image.
*   **Supervisor-in-the-Loop UI:** An intuitive interface for an expert supervisor to review, validate, and weight individual user annotations.
*   **Empirical Consistency Score:** Automatically calculates a consistency score for each rater by comparing their annotations on original vs. quality-control (QC) duplicate images.
*   **Dynamic Weighting:** Incorporates a-priori user expertise, calculated consistency, and supervisor-provided weights directly into the DS model.
*   **Certainty-Based Workflow:** Focuses supervisor attention on ambiguous images (i.e., those with mid-range certainty scores) while allowing a manual override for any image.
*   **Advanced Visualization:**
    *   Dual-canvas view to compare the original image with user annotations.
    *   Toggleable heatmap view to visualize the density and intensity of annotations.
    *   Ability to toggle individual user drawings on and off.
*   **Comprehensive Reporting:** Generates detailed CSV reports on the final consensus for each image and the performance (confusion matrix, effective weight) of each rater.

## File Structure

For the script to function correctly, your project must follow this directory structure:

```
.
├── phd_supervisor.py         # The main application script
│
├── image_dataset/            # Root folder for all images
│   ├── site_A/
│   │   ├── image_01.jpg
│   │   └── image_02.jpg
│   └── site_B/
│       ├── image_03.jpg
│       └── image_03_qc.jpg   # A quality-control duplicate of image_03
│
└── annotations/              # Folder for all data files
    ├── cropmarks_user1_5.xlsx  # Annotation file for 'user1' with expertise '5'
    ├── cropmarks_user2_3.xlsx  # Annotation file for 'user2' with expertise '3'
    │
    ├── supervisor_validations.json # (Generated) Stores supervisor's decisions
    ├── raters_report.csv           # (Generated) Rater performance metrics
    └── consensus.csv               # (Generated) Final consensus for each image
```

**Annotation File Naming:**
The user annotation files must follow the pattern `cropmarks_{USERNAME}_{EXPERTISE_SCORE}.xlsx`. The expertise score is optional (e.g., `cropmarks_user3.xlsx`).

## Getting Started

### Prerequisites

This script requires Python 3.x and the following libraries:

```
opencv-python
openpyxl
Pillow
pandas
numpy
```

You can install them all using pip:
```bash
pip install opencv-python openpyxl Pillow pandas numpy
```

### Running the Application

To run the application, navigate to the script's directory in your terminal and execute it with Python:

```bash
python phd_supervisor.py
```

## Usage and Workflow

The application is designed to facilitate an efficient review process for a supervisor.

### 1. Main Interface
*   **Top Panel:** Displays the current site, image date, and navigation controls. You can jump directly to a site using the dropdown menu.
*   **Left Canvas:** Shows the original, unaltered satellite image.
*   **Right Canvas:** Shows the same image with annotations overlaid. Use the **"Toggle Heatmap"** button or the **Spacebar** to switch between individual user drawings and a heatmap view.
*   **DS Certainty Panel:** Displays the model's current calculated "Certainty of Presence" for the image, along with the probabilities for 'Faint' and 'Clear' marks.
*   **Supervisor Validation Panel:** This is the primary workspace for validation.

### 2. Supervisor Validation Workflow
The system focuses your attention on images where the model is uncertain (e.g., certainty between 15% and 85%).

1.  **Review Annotations:** For each user who annotated the image, a column is displayed showing their name, their score, and their drawing (visible on the right canvas). You can click a user's name to toggle their drawing on/off.
2.  **Make an Initial Choice:**
    *   **Agree:** You concur with the user's assessment.
    *   **Disagree:** You do not concur with the user's assessment.
    *   **Unsure:** You are not certain. This applies a neutral weight of 1.0 and finalizes the validation immediately.
3.  **Provide Justification:**
    *   After choosing "Agree" or "Disagree," a context-specific checklist appears.
    *   For example, if you **disagree** with a user's mark (a "False Positive"), you will be asked to select the reason (e.g., "Machinery tracks," "Soil texture").
    *   If you **agree** with a user's mark, you can select "Drivers" that support the observation.
4.  **Submit Validation:** Click **"Submit"**. The system calculates a new weight for that user's observation based on your input, saves it, and re-runs the DS model to update the consensus.
5.  **Reset:** You can click the **"Reset"** button at any time to undo your validation for a specific user on that image.

### 3. Manual Override
If an image has a high or low certainty score, the validation controls are hidden. To review it anyway, click the **"Force Manual Validation"** button. This will reveal the validation controls for all users on that image.

## Methodology Deep Dive

#### Rater Consistency Calculation
To provide an empirical measure of rater performance, the system calculates a **consistency score**. This is achieved by analyzing annotations on pre-defined quality-control (QC) image pairs (e.g., `image_01.jpg` and `image_01_qc.jpg`). A "match" is counted if a user gives the same fundamental assessment for both images in a pair:
*   Both are marked as 'absence' (score 0).
*   Both are marked as 'presence' (score 1 or 2).

The consistency score is the ratio of matches to the total number of QC pairs a user has annotated. This score is then used as a multiplier for the user's base weight within the DS model.

#### Dawid-Skene (DS) Model
The core of the system is an implementation of the Dawid-Skene model, which uses an Expectation-Maximization (EM) algorithm. It treats the "true" label of an image as a hidden variable and iteratively performs two steps:
1.  **E-Step:** It estimates the probability of each image having a "true" label (None, Faint, or Clear), based on the current error profiles of the raters.
2.  **M-Step:** It uses these probabilities to re-calculate the error profile (or "confusion matrix") for each rater.

This process allows the model to learn which raters are more trustworthy and down-weight the influence of those who are less consistent, ultimately producing a robust probabilistic consensus.

#### Supervisor Weight Adjustment Model
When a supervisor validates an annotation, the system adjusts the weight of that specific observation using a simple and interpretable additive model.
*   Every observation starts with a baseline weight of **1.0**.
*   Based on the supervisor's checklist selections, predefined positive or negative **deltas** are added to this baseline.
    *   Example: Tagging a mark as a "False Positive" due to "Fairy rings" might apply a delta of **-0.5**.
    *   Example: Confirming a mark by tagging it as a "Driver" like "Aligned with known ditch plan" might apply a delta of **+0.4**.
*   The final weight is clamped between **0.1** and **1.5** to prevent any single validation from having an outsized effect on the model.

## Output Files

The application generates and maintains several output files in the `annotations/` directory:

*   `supervisor_validations.json`: A JSON file that stores all supervisor decisions. This file acts as the memory of the system, recording every choice, tag, and calculated weight for each validated annotation.
*   `raters_report.csv`: A detailed report on every rater, including their initial expertise, calculated consistency, final effective weight, and their full `3x3` confusion matrix as determined by the DS model.
*   `consensus.csv`: The final output for each image, containing the posterior probabilities for each label (P_Faint, P_Clear), the final aggregated P_Present, and the binary DS_Present decision.