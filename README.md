# Official Repo of ClaimSpect
![profile](asset/profile.png)

Official implementation for [ACL 2025](https://2025.aclweb.org/) main track paper [Beyond True or False: Retrieval-Augmented Hierarchical Analysis of Nuanced Claims](https://openreview.net/forum?id=6Io5Pmuh19).

## 📦 Repo Setup

1. Clone the repository:
```bash
git clone https://github.com/pkargupta/claimspect.git
cd ClaimSpect
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Data Construction

The data construction process is implemented in the `data/dtra` and `data/vaccine` directory. This process involves:



## 🔍 Claim Analysis

To run the claim analysis experiments:

```bash
bash script/run_experiments.sh
```

This script will execute the main claim analysis pipeline.

## 📈 Evaluation

We provide multiple evaluation scripts for different aspects of the system:

### Baseline Evaluation
Run the baseline evaluation scripts in `eval/baseline/` directory.

### LLM-as-Judge Evaluation
```bash
bash eval/eval_dtra.sh
```

### Retrieval Corpus Relevance Check
```bash
python eval/claim_examine/main.py
```

### Human Judge Evaluation
```bash
python eval/human_judge/main.py
```

### Human-Machine Alignment Evaluation
```bash
python eval/human_machine_align.py
```

## 📖 Citations
Please cite the paper and star this repo if you use ClaimSpect and find it interesting/useful, thanks! Feel free to open an issue if you have any questions.

```bibtex
@inproceedings{
    anonymous2025beyond,
    title={Beyond True or False: Retrieval-Augmented Hierarchical Analysis of Nuanced Claims},
    author={Anonymous},
    booktitle={Submitted to ACL Rolling Review - February 2025},
    year={2025},
    url={https://openreview.net/forum?id=6Io5Pmuh19},
    note={under review}
}
```