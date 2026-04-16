# Robust Opinion Maximization

This repository contains the anonymized source code to reproduce the algorithms, experiments, and figures presented in the paper Robust Opinion Maximization. 

We have included two mid-sized network datasets (`Epinions` and `Gowalla`) so the scripts can be tested immediately without needing to download external files. If needed tests on larger networks can be applied by downloading graphs from SNAP.

## 📂 Repository Structure

* `src/`: Contains the core algorithmic implementations (`core.py`) and the execution scripts for each Research Question (`rq1.py`, `rq2.py`, `rq3.py`). It also includes the `graph_loader.py` data pipeline.
* `data/`: 
  * `raw/`: Stores the raw `.gz` network files from SNAP.
  * `processed/`: Stores the heavily compressed `.npz` sparse matrices. Contains Epinions and Gowalla for instant out-of-the-box testing.
* `results/`: The output directory where the generated `.pdf` figures and LaTeX tables are automatically saved.
* `requirements.txt`: The Python dependencies required to run the code.

## ⚙️ Environment Setup

This code was developed and tested on Python 3.10+. To set up the environment, install the required packages:

```bash
pip install -r requirements.txt
