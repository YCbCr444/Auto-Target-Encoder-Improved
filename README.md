# Auto Target Encoder
![License](https://img.shields.io/badge/license-MIT-green)
![Python Version](https://img.shields.io/badge/python-3.8+-blue)
![Windows](https://img.shields.io/badge/-Windows-0078D6?logo=windows&logoColor=white)

**A sophisticated, GUI-based encoding tool designed for **automated batch processing** of your videos that do not require comprehensive fine-tuning. It leverages machine learning to create high-quality, efficient AV1 video encodes. This application automates the entire workflow for large batches of files: it learns from past encodes to predict optimal quality settings, intelligently analyzes each video's complexity, and displays the progress of all parallel jobs in a real-time dashboard.**

This tool moves beyond single-file, trial-and-error encoding by building persistent knowledge. A **RandomForest machine learning model** predicts the exact CQ/CRF value needed to hit a target quality score (VMAF, SSIMULACRA2, BUTTERAUGLI), while other models provide highly accurate ETA predictions by learning your hardware's real-world performance across hundreds of encodes.

<details>
  <summary><strong>What are Perceptual Metrics?</strong> (Click to expand)</summary>
  
  Perceptual quality metrics like **VMAF**, **SSIMULACRA2**, and **Butteraugli** are designed to estimate how a human viewer would perceive video quality. This script uses them as a target to ensure encodes are not just mathematically similar, but *visually* excellent, achieving the perfect balance between file size and quality.
</details>

</div>

---

## 🎯 Key Features

### 🧠 Machine Learning Core for Batch Optimization

* **Predictive Quality Model:** Instead of a slow, brute-force search for every file in your batch, the application uses a `RandomForestRegressor` model to instantly predict the optimal CQ/CRF value needed to achieve your target quality score. This drastically reduces total analysis time.
* **Adaptive Error Correction:** The system learns from its prediction errors during a batch. If a prediction is off, it analyzes the mistake and applies a correction factor to future predictions for similar content, making the model progressively more accurate as the queue progresses.
* **Data-Driven ETA Predictions:** Forget simple estimates. The ETA is calculated for the **entire remaining queue** by combining predictions from multiple ML models that analyze every stage of the process: sample creation time, quality search time, and the final encoding speed (FPS) based on your specific hardware.
* **Persistent Learning Database:** All performance and quality results are stored in a local SQLite database. This data is used to automatically train and improve the ML models every time you launch the application, making it smarter and more accurate with every batch you process.

### 💻 Modern GUI for Batch Control

* **Real-Time Dashboard:** Monitor multiple parallel encodes at once. Each worker has its own panel displaying the current filename, a live progress bar, and a detailed log output, giving you a complete overview of your batch.
* **Comprehensive Settings Editor:** A user-friendly, tabbed window allows you to configure every aspect of the encoding process, from file paths and encoder settings to quality targets and sampling methods, with helpful tooltips for every option.
* **Efficient Batch Management:** Built from the ground up for batch encoding. Scan entire directory trees, add hundreds of files to the queue, and let the application process them automatically. A dedicated "Completed" tab provides a detailed report of all finished jobs.

<p align="center">
  <img src="https://github.com/Snickrr/Auto-Target-Encoder/blob/main/demo.gif" alt="Live Demo of Auto Target Encoder">
</p>

### 🎯 Advanced Quality Targeting

* **Multi-Metric Support:** Target your preferred perceptual quality metric: **VMAF** (both average and percentile), **SSIMULACRA2**, or **Butteraugli**.
* **Intelligent Tiered Sampling:** A robust, three-stage analysis system ensures reliable results for any video type:
    * **Tier 1:** High-speed scene detection using FFmpeg.
    * **Tier 2:** "SmartFrames" analysis of keyframe density and temporal complexity.
    * **Tier 3:** A fallback to evenly-spaced time intervals for low-complexity or unusual videos.
<details>
  <summary><strong>What is "SmartFrames" sampling?</strong> (Click to expand)</summary>
  
"SmartFrames" is an intelligent video sampling method designed to select the most representative clips for quality testing before a full encode. Its goal is to find short segments that represent the most complex or visually demanding parts of a video, ensuring the final quality settings are robust enough to handle the toughest scenes.

It works in a four-step process:

1.  **Keyframe Extraction**: First, it performs a high-speed scan of the video to identify the timestamp of every single keyframe.
2.  **Complexity Scoring**: Each keyframe is then assigned a "complexity score." A keyframe gets a higher score if it's part of a high-action sequence (meaning it's surrounded by many other keyframes) or indicates a very quick scene change.
3.  **Temporal Bucketing**: The video's timeline is divided into a number of equal segments, or "buckets." For example, if four samples are needed, the video's duration is split into four equal time slots.
4.  **Best-of-Bucket Selection**: Finally, the system looks inside each time bucket and selects the single keyframe with the highest complexity score from that segment.

The result is a set of sample points that are both evenly distributed throughout the video's duration and representative of its most challenging moments. This leads to a more accurate and reliable quality assessment than just picking scenes at random or at fixed intervals.

</details>

### ⚙️ Powerful & Flexible Encoding

* **Dual AV1 Encoder Support:**
    * **NVENC AV1:** Fast, hardware-accelerated encoding on NVIDIA GPUs.
    * **SVT-AV1:** High-efficiency, CPU-based encoding for maximum quality and compression.
* **Advanced Color Handling:** Full preservation of source color data `(color_space, color_primaries, color_transfer)`, ensuring your encodes look exactly as intended.
* **10-bit Pipeline:** Supports 10-bit output to prevent banding and preserve color fidelity, which is ideal for the AV1 codec.

### 📂 Smart File Management for Automation

* **Pre-emptive Filtering:** A fast pre-scan can automatically skip files based on custom rules (e.g., minimum duration, file size, or resolution-specific bitrates) *before* wasting time on a full analysis, making large batches more efficient.
* **Set-and-Forget Automation:** Combine the powerful pre-filtering, queue management, and smart encoding logic to process huge batches of files with minimal intervention. It intelligently handles skips, failures, and successes on its own.
* **Configurable I/O:** Full control over output directories, custom file suffixes, and optional automatic deletion of source files upon success.

<details>
  <summary><strong>❗ML Model Training & Fallback Behavior</strong> (Click to expand)</summary>
  
### How the Machine Learning Activates

The machine learning features are not active on the first run. The application needs to "learn" from your hardware and settings by gathering data from completed encodes.

* **ETA & Performance Models:** These models typically begin to activate and provide accurate predictions after the application has successfully logged around **15-20 encodes**.
* **Quality Prediction Model:** This model is more data-intensive. It becomes effective at predicting the optimal CQ/CRF value after it has logged approximately **50 quality data points** for a *specific combination* of settings (e.g., for NVENC with VMAF, or SVT-AV1 with SSIMULACRA2).

**In short: The more you use the application, the smarter, faster, and more accurate it becomes.**

<br>

### How the Script Works Without ML

The application is **fully functional** even before the ML models are trained. It simply uses more traditional, robust methods as a fallback:

* **For Quality Searching:** Instead of predicting the best quality setting in one shot, the script uses a reliable **interpolation search algorithm**. It intelligently tests a few different quality values to methodically narrow down the range and find the one that meets your target score. This process is slower than the ML prediction but is guaranteed to be accurate.
* **For ETA Predictions:** Before the performance model is trained, ETAs are based on **simple heuristics** (basic formulas that factor in video resolution, duration, and encoder type). These estimates are less precise than the ML predictions but still provide a general idea of the time required. However, please note that less work was put into perfecting this model as it is NOT the target of this project. 

</details>

<details>
  <summary><strong>💡 How the ML-Accelerated Quality Search Works</strong> (Click to expand)</summary>
  
The script's primary goal is to find the highest CQ/CRF value (for the best compression) that still meets your quality target. To do this as fast as possible, it uses a unique, confidence-based hybrid strategy that blends machine learning with traditional search methods.

Here’s how it works:

1.  **ML Prediction & Confidence Score**
    The process begins when the trained Quality Model analyzes the video's features. It doesn't just predict a single CQ value; it also returns a **confidence level** for its own prediction (High, Medium, or Low).

2.  **A Strategy for Every Situation**
    The script's next action depends entirely on that confidence level:

    * **High Confidence:** The script trusts the model and takes an aggressive, "fast-track" approach. It assumes the prediction is very close to correct and only performs one or two quality tests right around that value to confirm. This is the quickest path to a successful result.

    * **Medium Confidence:** The script is cautiously optimistic. It tests the predicted value first. If the result isn't perfect, it uses that new data point to perform a very narrow and targeted search, saving significant time compared to a full search.

    * **Low Confidence (or No ML Model):** The script plays it safe. It knows the prediction might be unreliable, so it falls back to the robust and traditional **interpolation search algorithm**. This method is slower but methodically narrows down the options to guarantee it finds the correct quality setting.

In essence, this system combines the raw speed of machine learning with the guaranteed accuracy of a methodical search. It only relies on the ML prediction when it's confident, ensuring both speed and reliability.

</details>

<details>
  <summary><strong>⚠️ Important: On Source File Deletion</strong> (Click to expand)</summary>
  
The setting `delete_source_file` is a **destructive feature** that should be used with extreme caution.

When this option is enabled in the settings:
* The original source file will be **permanently deleted** from your system after a successful encode.
* Deletion only occurs if the new file is successfully created, verified, and meets the minimum size reduction threshold. It will not delete the source if the encode fails or is skipped.

It is **strongly recommended** that you run the script on a small batch of test files first to ensure everything works as expected before enabling this feature on your main library. **Always have backups of important media.** This feature is disabled by default for your safety.

</details>

<details>
  <summary><strong>⚙️ Resilience & Reliability: Other Built-in Fallback Mechanisms</strong> (Click to expand)</summary>
  
Beyond the ML-to-traditional fallbacks, the script includes several other automatic systems designed to handle problematic videos and unexpected errors gracefully.

### Intelligent Sampling Fallback System

The script needs to select sample clips from every video for analysis, but not all videos are structured the same way. To handle this, it uses a tiered fallback system to guarantee a successful analysis every time.

* **Attempt #1: Tier 1 (Scene Detection)**
    The script first tries the fastest method: using FFmpeg to detect distinct scene changes. This is ideal for movies and TV shows. However, it can fail on content with very long, static shots like presentations or gameplay videos.

* **Attempt #2: Tier 2 (SmartFrames)**
    If Tier 1 fails to find enough scene changes, the script **automatically falls back** to the more robust SmartFrames analysis. This method analyzes keyframe density and is more reliable, but slightly slower.

* **Guaranteed Success: Tier 3 (Time Intervals)**
    If a video is highly unusual and even SmartFrames fails (e.g., a screen recording with no keyframes), the script **falls back a final time** to a foolproof method: selecting clips at simple, evenly-spaced intervals.

This tiered cascade ensures that *every video* can be successfully analyzed for quality testing, regardless of its content.

<br>

### Final Encode Safeguards

The final, full-length encode is the most time-consuming part of the process. To prevent the script from getting stuck for hours on a single problematic file, it uses several safeguards:

* **Stall & Freeze Detection:** The script actively monitors the FFmpeg process. If the output file stops growing in size or if the progress bar freezes for an extended period, the script will automatically terminate the stalled encode and mark it as failed, allowing the batch queue to move on.

* **Post-Encode Verification:** After an encode finishes, the script doesn't just assume it worked. It performs a final, quick check on the output file to ensure it's not corrupt, is readable, and has the correct video duration. If this check fails, the faulty file is discarded and the process is logged as a failure.

These mechanisms prevent the entire batch process from being halted by a single faulty video and ensure that you never end up with silent, corrupted files in your output directory.

</details>

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed and accessible:

* ✅ **Python 3.8+**: Make sure it's added to your system's PATH. ([Download](https://python.org/downloads/))
* ✅ **FFmpeg**: The latest builds of `ffmpeg.exe` and `ffprobe.exe` are required. Full builds for Windows are available from [Gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
* ✅ **FFVShip (Optional)**: Required if you want to use the **SSIMULACRA2** or **Butteraugli** quality metrics. Requires compatible NVIDIA or AMD GPUs. You can find releases on the [VShip GitHub page](https://github.com/Line-fr/Vship/releases).

### Installation & Configuration

1.  **Clone or Download the Repository**
    Click the green "Code" button on the GitHub page and select "Download ZIP". Unzip the folder to a permanent location on your computer.

2.  **Install Python Libraries**
    Open a Command Prompt or Terminal and run the following command to install the necessary libraries:
    ```bash
    pip install customtkinter scikit-learn psutil numpy
    ```

3.  **Launch the Application**
    Run the `Auto Target Encoder` script.
    ```bash
    python "Auto Target Encoder.pyw"
    ```

4.  **Configure Settings via the GUI**
    * On first launch, click the **Settings** button in the GUI and configure your paths and desired settings.
    * You are now ready to scan a directory and start your batch encode!

---

## 🤝 Contributing

This project was created by someone with no prior coding experience, using AI assistance (Claude Opus 4.1 and Gemini 2.5 Pro) for advanced mathematics and coding implementation. The core ideas and extensive debugging/fine-tuning were done manually.

Contributions are welcome! Please feel free to:
* Report bugs or suggest features by opening an issue. I will do my best to fix bugs and implement new features with the help of AI. 

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.


<br>
<div align="center">

**Star this repository if it helped you automate your encoding workflow! ⭐**


</div>




