# ONLY BY HANDS ✋💻
> *Just with hands. No mouse, no keyboard.*

**OnlyByHands** is an innovative application that leverages Computer Vision and Machine Learning to transform your hands into a complete computer controller. Featuring a Cyberpunk-inspired GUI and real-time gesture tracking, this project delivers a futuristic, hands-free interaction experience straight out of a sci-fi movie.

---

## 🌟 Core Features

*   **AR Ink Overlay:** Draw and write directly in the air. The virtual ink is overlaid seamlessly onto the camera feed without generating pop-up windows, ensuring you never lose cursor focus when typing in other applications (like Chrome or Word).
*   **"Vision Pro" Mouse Mechanism:** Move the cursor using your index finger and click by pinching your thumb and index finger together. This design completely eliminates the occlusion issue (where fingers block the camera's view) present in traditional gesture control systems.
*   **Anti-Jitter System:** Integrates an Exponential Moving Average (EMA) filter, a Velocity Clamp (max 80px), and a 3-pixel Deadzone. This combination prevents random cursor jumps and provides a smooth, stable experience identical to a physical mouse.
*   **Cross-Toggle Safety:** Switch between Mouse and Keyboard modes seamlessly. A 1.5-second freeze cooldown mechanism ensures safety by locking inputs during the transition, preventing accidental clicks.
*   **Handwritten OCR AI:** Integrated neural network trained on the EMNIST dataset to recognize handwritten characters drawn in the air, instantly typing them via a virtual keyboard controller.

---

## 🕹️ Gesture Mapping

### 1. System Toggles (Mode Switching)
| Hand | Gesture | Action |
| :--- | :--- | :--- |
| **Left** | 🤘 Rock (Spider-man) hold for 0.8s | Toggle Mouse Mode ON/OFF |
| **Right** | 🤘 Rock (Spider-man) hold for 0.8s | Toggle Drawing/Keyboard Mode ON/OFF |

### 2. Drawing & Keyboard Mode
*Active only when Keyboard Mode is ON.*

| Hand | Gesture | Action |
| :--- | :--- | :--- |
| **Left** | 🤏 Pinch (Index & Thumb) | Draw virtual ink in the air |
| **Right** | 👌 OK Sign | Activate OCR to read the drawn character |
| **Right** | 👍 Thumbs Up | Press `Space` key |
| **Right** | ✌️ Number 2 (L-shape) | Press `Enter` key |
| **Right** | 👎 Thumbs Down | Press `Backspace` key |
| **Right** | 🖐️ High-Five (Open palm) | Clear the virtual whiteboard |

### 3. Mouse Mode
*Active only when Mouse Mode is ON.*

| Hand | Gesture | Action |
| :--- | :--- | :--- |
| **Left** | ☝️ Move Index Finger Tip | Move the mouse cursor |
| **Left** | 🤏 Pinch (Index & Thumb) | Left-click |
| **Right** | 🤏 Pinch & Drag Up/Down | Smooth scrolling |

---

## ⚙️ Tech Stack

*   **Language:** Python 3
*   **Computer Vision:** OpenCV, MediaPipe (Hand Landmarker API)
*   **Machine Learning:** TensorFlow/Keras (Optimizing neural networks for handwritten text recognition and image matrix processing).
*   **GUI:** CustomTkinter (Dark/Neon theme), Multi-Threading.
*   **System Control:** Pynput, PyAutoGUI.

---

## 🚀 Installation & Usage

### For End-Users (Setup.exe)
1. Download the latest `OnlyByHands_Setup.rar` from the **Releases** section.
2. Unzip this file, and run  `OnlyByHands_Setup.exe`. 
3. Select your installation directory (e.g., `C:\Program Files`), and check the option to create a Desktop Shortcut.
4. Open the app from your Desktop, click **START INITIALIZATION** on the interface, and enjoy.

### For Developers (Run from Source)
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/doannaman/OnlyByHands.git](https://github.com/doannaman/OnlyByHands.git)
2. Install the required dependencies: pip install -r requirements.txt
3. Ensure that the core assets (emnist_model.h5 and hand_landmarker.task) are placed in the root directory.

---

## 🤝 Contributing
Feel free to open a Pull Request or an Issue to discuss details. Please ensure that all code maintains cross-platform compatibility and protects the main GUI thread from blocking operations.

---

## Acknowledgments & Third-Party Assets

This project makes use of the following third-party models and datasets:

* **Hand Tracking Model (`hand_landmarker.task`)**: 
  * Provided by [Google MediaPipe](https://developers.google.com/mediapipe).
  * Distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

* **EMNIST Character Recognition Model (`emnist_model.h5`)**: 
  * Custom-trained model based on the **EMNIST Balanced** dataset provided via TensorFlow Datasets.
  * **Dataset Citation**: 
    > Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). *EMNIST: an extension of MNIST to handwritten letters*. arXiv preprint arXiv:1702.05373. Available at: [https://arxiv.org/abs/1702.05373](https://arxiv.org/abs/1702.05373)
