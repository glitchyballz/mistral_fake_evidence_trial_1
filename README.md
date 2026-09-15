# LLM Behaviour Under User Pressure

Experimental framework for investigating LLM behaviour under user pressure, comparing fake evidence and unsupported persistence across Mistral and Magistral using MMLU and GSM8K-Platinum.

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support recommended
- CUDA-compatible PyTorch
- 4-bit quantization with BitsAndBytes

Install the required libraries:

```bash
pip install torch transformers accelerate bitsandbytes datasets mistral-common
```

The experiment was developed and tested with:

- Python 3.11
- PyTorch 2.6+
- Transformers 4.57+
- Accelerate 1.10+
- BitsAndBytes 0.46+
- Datasets 4.1+
- mistral-common 1.8+

## Phase 2

Each team member runs one model × technique cell using `phase2_trial_loop.py`.

Before running the file, open:

```
phase2_trial_loop.py
```

and set:

```python
MODEL_KEY = "mistral"
TECHNIQUE = "fake_evidence"
```

Use the assigned combination:

| Team member | Model | Technique |
|---|---|---|
| Haque | Mistral | Fake Evidence |
| Nikhil | Mistral | Persistence |
| Cencen | Magistral | Fake Evidence |
| Thuyen | Magistral | Persistence |

For example, the Magistral + Fake Evidence cell is:

```python
MODEL_KEY = "magistral"
TECHNIQUE = "fake_evidence"
```

Then run:

```bash
python phase2_trial_loop.py
```

Each cell runs:

- 50 questions
- 2 pressure strategies
- 3 replications
- 300 trials total

The output will be saved automatically under:

```
phase2_results/
```

## Important

Do not modify the prompts, pressure assignments, question selection logic, or experimental settings unless instructed. Each team member should only change `MODEL_KEY` and `TECHNIQUE` to their assigned combination.
