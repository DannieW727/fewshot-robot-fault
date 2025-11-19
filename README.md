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
```

## citation
```bash
@ARTICLE{11202171,
  author={Wang, Danyi and Wang, Tianyi and Wang, Xiaoya},
  journal={IEEE Access}, 
  title={Few-Shot Fault Diagnosis for Industrial Robot Transmission Systems via a Prototypical Time-Frequency Mixer}, 
  year={2025},
  volume={13},
  number={},
  pages={178045-178059},
  keywords={Fault diagnosis;Industrial robots;Vibrations;Time-frequency analysis;Feature extraction;Sensors;Robustness;Metalearning;Time-domain analysis;Prototypes;Fault diagnosis;few-shot learning;industrial robot;prototypical time-frequency mixer;transmission system},
  doi={10.1109/ACCESS.2025.3620386}}
```
