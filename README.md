# Proton Therapy Laboratory Portfolio ☢️

Welcome to my Proton Therapy Portfolio! This repository contains a collection of advanced laboratory assignments and projects completed during the "Engineering challenges in protontherapy" (LGBIO 2070) course at UCLouvain. I attended this course as part of my Erasmus exchange program.

The projects focus on the core physical and computational aspects of proton therapy, ranging from particle transport simulations to beam optics optimization and treatment planning using real patient data.

## 📂 Structure of the Repository

The portfolio is divided into three main modules:

### 1️⃣ Lab 1: Dose Calculation (`Lab1_Dose_Calculation`)
* **Focus:** Implementation of a **Monte Carlo simulation** from scratch to model proton dose deposition in biological tissues.
* **Key Concepts:** Continuous Slowing Down Approximation (CSDA), Energy Straggling, Multiple Coulomb Scattering, and Nuclear Interactions (elastic and inelastic proton-proton and proton-oxygen collisions).
* **Contents:** Python script simulating depth-dose curves (Bragg Peak) in homogeneous and heterogeneous phantoms (water/compact bone), raw data, and a comprehensive Technical Report analyzing the results against MCsquare reference data.

### 2️⃣ Lab 2: System Design (`Lab2_System_Design`)
* **Focus:** Design and simulation of a proton therapy delivery system for lung cancer, including beam production, transport, and delivery.
* **Key Concepts:** 
  * **Beam Production:** Isochronous cyclotron parameter calculation (magnetic fields, frequencies, high voltage).
  * **Beam Transport:** Phase-space beam propagation and optimization of quadrupole focal lengths and drift distances using `scipy.optimize` to strictly meet safety and symmetry constraints.
  * **Beam Delivery & Interplay Effects:** Simulation of 34x34 pencil beam scanning on a moving target (modeling patient breathing) and implementation of an **Adaptive Spot Triggering** strategy to mitigate interplay effects.
* **Contents:** Python simulation code (`Lab2_Beam_Simulation_main.py`) and a detailed Technical Safety Report.

### 3️⃣ Lab 3: Treatment Planning (`Lab3_Treatment_Planning`)
* **Focus:** Handling and processing of clinical imaging data for treatment planning optimization.
* **Key Concepts:** **DICOM** data parsing, Hounsfield Units (HU) conversion, 3D CT scan visualization (Axial, Coronal, Sagittal planes), and analysis of image distortions (e.g., metal artifacts).
* **Contents:** An interactive Jupyter Notebook (`Lab3_Treatment_Planning_main.ipynb`) demonstrating the data processing pipeline.

## 🛠️ Technologies & Tech Stack
* **Languages:** Python
* **Environments:** Jupyter Notebook
* **Scientific & Data Analysis:** `NumPy`, `SciPy` (Optimization algorithms)
* **Medical Imaging:** `pydicom` (DICOM format manipulation)
* **Data Visualization:** `Matplotlib` (Dose maps, phase-space plots, depth-dose curves)

## 🎯 Key Skills Demonstrated
* Medical Physics & Particle Tracking (Monte Carlo)
* Mathematical Optimization & Constraint Solving
* Signal Processing & Motion Mitigation Algorithms
* Medical Imaging Data Handling (CT/DICOM)
* Technical Writing & Reporting
