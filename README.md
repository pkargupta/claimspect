# Official Repo of ClaimSpect
![profile](asset/profile.png)

Official implementation for [ACL 2025](https://2025.aclweb.org/) main track paper [Beyond True or False: Retrieval-Augmented Hierarchical Analysis of Nuanced Claims](https://openreview.net/forum?id=6Io5Pmuh19).

## 🪧 Paper Abstract

Claims made by individuals or entities are oftentimes nuanced and cannot be clearly labeled as entirely "true" or "false"---as is frequently the case with scientific and political claims. However, a claim (e.g., "vaccine A is better than vaccine B") can be dissected into its integral aspects and sub-aspects (e.g., efficacy, safety, distribution), which are individually easier to validate. This enables a more comprehensive, structured response that provides a well-rounded perspective on a given problem while also allowing the reader to prioritize specific angles of interest within the claim (e.g., safety towards children). Thus, we propose ClaimSpect, a retrieval-augmented generation-based framework for automatically constructing a hierarchy of aspects typically considered when addressing a claim and enriching them with corpus-specific perspectives. This structure hierarchically partitions an input corpus to retrieve relevant segments, which assist in discovering new sub-aspects. Moreover, these segments enable the discovery of varying perspectives towards an aspect of the claim (e.g., support, neutral, or oppose) and their respective prevalence (e.g., "how many biomedical papers believe vaccine A is more transportable than B?"). We apply ClaimSpect to a wide variety of real-world scientific and political claims featured in our constructed dataset, showcasing its robustness and accuracy in deconstructing a nuanced claim and representing perspectives within a corpus. Through real-world case studies and human evaluation, we validate its effectiveness over multiple baselines.

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

1. **Claim Construction**: Generate initial claims using `data/dtra/raw_claims/generate_claims.py`
2. **Literature Searching**: Search for relevant papers using `data/dtra/get_literature/get_literature_meta_info.py`
3. **Literature Download**: Download paper content using `data/dtra/get_literature/get_literature_body_from_url.py`
4. **Literature Chunking**: Split papers into manageable chunks using `data/dtra/chunking/run_chunking.sh`

Each step builds upon the previous one to create a comprehensive dataset for claim analysis.


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
