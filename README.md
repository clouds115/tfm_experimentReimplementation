# On the Evaluation of DNN-Based Automatic Commit Message Generation: Replication and Experimental Reassessment
EMSE Master's Thesis Project - Job Scripts Repository
Claudio Vincenzo Catalano Leiva - June 2026

This repository contains the job scripts for the second part of the Master's Thesis project's reimplementation efforts. The experiments are executed on a supercomputer using SLURM job scheduling system.

## Project Overview

The project implements two main experiments with a Latin Square experimental design to control for order effects in machine translation model training and inference.

## Repository Structure

```
tfm_experimentReimplementation/
├── exp1_mcmd/              # Experiment 1: MCMD Training & Inference
│   └── jobs/
│       ├── LS1/            # Latin Square Side 1
│       │   └── EXP1_LS1_RUN[1-5].slurm
│       └── LS2/            # Latin Square Side 2
│           └── EXP1_LS2_RUN[1-5].slurm
├── exp2_mcmd-nl/           # Experiment 2: MCMD-NL Training & Inference
│   └── jobs/
│       ├── LS1/            # Latin Square Side 1
│       │   └── EXP2_LS1_RUN[1-5].slurm
│       └── LS2/            # Latin Square Side 2
│           └── EXP2_LS2_RUN[1-5].slurm
└── evaluate_simple.py      # Evaluation script
```

## Experiments

### Experiment 1 (MCMD)

**LS1 (Latin Square Side 1):**
- Training: CCT5 → COME
- Inference: MCMD and MCMD-NT

**LS2 (Latin Square Side 2):**
- Training: COME → CCT5 (swapped order)
- Inference: MCMD and MCMD-NT

### Experiment 2 (MCMD-NL)

**LS1 (Latin Square Side 1):**
- Training: MCMD-NL
- Inference: MCMD-NL

**LS2 (Latin Square Side 2):**
- Training: MCMD-NL (swapped order)
- Inference: MCMD-NL

## Running Jobs

To execute a job on the supercomputer, use the SLURM batch command:

```bash
sbatch path/to/job_file.slurm
```

Example:
```bash
sbatch exp1_mcmd/jobs/LS1/EXP1_LS1_RUN1.slurm
```

Each experiment has 5 runs per Latin Square side for statistical robustness.

## Results

Links to the result repositories will be posted here:

- **Experiment 1 (MCMD) LS1 Results:** [https://huggingface.co/clouds125/TFM_EXP1_MCMD_LS1/tree/main]
- **Experiment 1 (MCMD) LS2 Results:** [https://huggingface.co/clouds125/TFM_EXP1_MCMD_LS2/tree/main]
- **Experiment 2 (MCMD-NL) LS1 Results:** [https://huggingface.co/clouds125/TFM_EXP2_MCMD-NL_LS1/tree/main]
- **Experiment 2 (MCMD-NL) LS2 Results:** [https://huggingface.co/clouds125/TFM_EXP2_MCMD-NL_LS2/tree/main]
