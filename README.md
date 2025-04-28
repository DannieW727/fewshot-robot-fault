# PTFM: Prototypical Time-Frequency Mixer for Few-Shot Fault Diagnosis

This repository provides the implementation of **Prototypical Time-Frequency Mixer (PTFM)**,  
a lightweight and effective framework designed for few-shot fault diagnosis in industrial robot transmission systems.

PTFM integrates **temporal modeling** via MLP-Mixer blocks, **frequency-domain enhancement** via FFT,  
and **prototypical networks** for metric-based classification, enabling robust fault diagnosis under extremely limited labeled data conditions.

##  Project Structure

```bash
PTFM-FewShot-FaultDiagnosis/
├── README.md
├── requirements.txt
├── scr/
│   ├── main.py          # main function
│   ├── main_CWRU.py          # main function
│   ├── main_JNU.py          # main function
│   ├── model.py          # model definition
│   ├── utils.py          # Helper functions
├── notebooks/
│   ├── IndustrialRobot.ipynb    # Few-shot diagnosis on Industrial Robot dataset
│   ├── CWRU.ipynb                # Few-shot diagnosis on CWRU bearing dataset
│   ├── JNU.ipynb                 # Few-shot diagnosis on JNU gearbox dataset
├── datasets/
