# Zomato_PS1
Zomato KPT Optimizer: Improving Zomato's Kitchen Prep Time accuracy via passive environmental telemetry (BLE/Acoustics), unforgeable Scan-to-Dispatch workflows, and a dynamic Merchant Reliability Index (MRI).
# 🍽️ Project Omni-Signal: High-Fidelity KPT Ground Truth Engine

## 📌 The Challenge
Zomato's delivery time accuracy relies heavily on Kitchen Prep Time (KPT). Currently, KPT models depend on merchants manually pressing a "Food Ready" (FOR) button. This introduces severe human and operational bias, blinding the system to actual kitchen load (competitor orders, dine-in rush) and resulting in high P90 ETA prediction errors and rider idle times.

## 🚀 Our Solution: Passive Telemetry & Algorithmic De-biasing
**Omni-Signal** shifts KPT prediction from *Active Input* (relying on manual merchant clicks) to *Passive Telemetry* (systematically measuring the environment). We introduce an architecture that enriches Zomato's existing models without requiring changes to their core ML infrastructure.

### 🔑 Key Pillars

#### 1. Zero-CapEx Edge Telemetry (Hardware-Assisted)
Instead of deploying new hardware, we turn the existing Zomato Merchant tablet into an environmental sensor:
* **Kitchen Chaos Index (Acoustic):** A lightweight local model categorizes ambient decibel levels to estimate kitchen intensity without recording speech.
* **Crowd Sniffing (BLE/Wi-Fi):** Periodically scans local device density to accurately proxy "hidden" dine-in and competitor rush loads.

#### 2. The Merchant Reliability Index (MRI)
An algorithmic "truth filter" that automatically penalizes merchants who game the system (e.g., marking food ready when it isn't to avoid penalties). 
* We calculate an order-level discrepancy score and maintain a rolling trust score ($\alpha$) for every merchant using an Exponential Moving Average (EMA).
* **Mathematical Override:** The system dynamically adjusts the predicted KPT based on a merchant's historical bias, filtering out noisy manual inputs.

#### 3. Unforgeable Ground Truth (Workflow Redesign)
* **Scan-to-Dispatch Protocol:** Replacing the manual "Food Ready" button with a mandatory QR scan of the printed Zomato receipt attached to the *sealed* bag. This guarantees a true, unforgeable timestamp for model training.

## 📊 Impact on Success Metrics
* **↓ P90 ETA Prediction Error:** Caught by environmental crowd-sniffing detecting unexpected Friday night dine-in rushes.
* **↓ Rider Idle Time:** MRI filters out low-trust merchants, preventing riders from arriving prematurely based on false FOR signals.
* **↑ Scalability:** Requires zero custom hardware; algorithms run with $O(1)$ state updates per merchant.

## 🛠️ Proposed Tech Stack
* **Algorithmic Core:** Python, NumPy, Pandas
* **Edge Telemetry Mock:** PyTorch Mobile / TensorFlow Lite
* **Backend Pipeline:** FastAPI
