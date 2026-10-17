# Math Problem Solving with LLMs: CoMAT Prompting, Shapley Analysis & GRPO Fine-Tuning

A set of experiments on **college-level math reasoning** with small open LLMs (Qwen2 / Qwen3) on the **MMLU-Redux College Mathematics** benchmark. The project:

1. Evaluates models with **CoMAT** (Chain of Mathematically Annotated Thought), a prompting method that has the model translate a problem into symbolic form before solving it step by step.
2. Measures how much each reasoning step contributes to accuracy using **Shapley values**.
3. Fine-tunes a small model with **GRPO** (Group Relative Policy Optimization) reinforcement learning, using a correctness-based reward.

## Repository Structure

| File | Description |
| --- | --- |
| `main.py` | CLI entry point: loads a Qwen model, runs CoMAT evaluation on the dataset, saves results and accuracy |
| `mmlu_redux.py` | Evaluation loop: formats options, queries the model, extracts the final answer letter (A–D) and computes accuracy |
| `utils.py` | `predict_model()` builds the chat template, generates a response (temperature / max-token config) and decodes only the new tokens |
| `CoMAT_Instruction.py` | The CoMAT system prompt (`INSTRUCTION`), built from `prompt-instruction.txt` |
| `prompt-instruction.txt` | Source text of the CoMAT prompt: a 7-step symbolic reasoning procedure with a worked example |
| `shapley_value_evaluation.py` | Computes the Shapley value of each of the 4 CoMAT reasoning steps from ablation results |
| `grpo_finetune.py` | GRPO fine-tuning of `Qwen2-0.5B-Instruct` with TRL, with model outputs saved before and after training |
| `mmlu-redux-college_mathematics_dataset.csv` | 100 multiple-choice college math questions (MMLU-Redux; questions with `error_type == "ok"` are evaluated) |
| `evaluation_with_steps.csv` | Step-ablation results: which reasoning steps were removed (`stepN_missing`) and whether the answer was correct |
| `requirements.txt` | Python dependencies |

## 1. CoMAT Evaluation (`main.py`)

The model gets a system prompt that requires it to:

1. define predicates, functions and variables,
2. parse the problem into logical rules,
3. state the facts explicitly,
4. parse the question into symbolic form,
5. solve it step by step,
6. derive the answer, and
7. only then match it to an option, ending with `Final Answer: X`.

The answer is extracted with a cascade of regular expressions: first `Final Answer: X`, then a standalone letter in the last 50 characters, then one in the last 20. Results are written incrementally to JSON.

```bash
python main.py --model qwen2 --temperature 0.1 --max_token_limit 2000
```

| Argument | Options | Default |
| --- | --- | --- |
| `--dataset` | `mmlu-redux-college_mathematics` | same |
| `--method` | `comat` | `comat` |
| `--model` | `qwen2` → `Qwen/Qwen2-1.5B-Instruct`, `qwen3` → `Qwen/Qwen3-1.7B` | `qwen2` |
| `--temperature` | float | `0.1` |
| `--max_token_limit` | int | `2000` |

Output goes to `final_results/<dataset>/<method>/<model>/`:

- `comat_<model>.json`: per-question model output, extracted answer, correct answer and correctness
- `comat_<model>_log.txt`: run configuration and final accuracy

## 2. Shapley Value Analysis (`shapley_value_evaluation.py`)

How much does each CoMAT step (1–4) actually help? Using the ablation data in `evaluation_with_steps.csv`:

1. Each row is mapped to the set of **missing** steps.
2. For every subset *S* of missing steps, **v(S)** is the mean accuracy of the rows with exactly that subset missing.
3. For every permutation of the steps, the **marginal contribution** of step *i* is accumulated as `v(S ∪ {i}) − v(S)`. Permutations with missing coalitions are skipped.
4. The **Shapley value** of each step is its average marginal contribution over the valid permutations.

```bash
python shapley_value_evaluation.py
```

## 3. GRPO Fine-Tuning (`grpo_finetune.py`)

- **Model:** `Qwen/Qwen2-0.5B-Instruct`
- **Data:** the MMLU-Redux CSV, formatted as `CoMAT instruction + question + A–D options`, with an 80/20 train/eval split (`seed=42`)
- **Reward:** `1.0` if the option letter found in the last 20 characters of a completion matches the correct answer, otherwise `0.0`. Rewards are computed for a group of completions per prompt, as GRPO requires.
- **Trainer:** TRL `GRPOTrainer` with default `GRPOConfig`
- **Outputs** (in `Qwen2-0.5B-GRPO/`):
  - `Qwen2-0.5B-base_OUTPUTS.json`: eval-set generations **before** training
  - `Qwen2-0.5B-GRPO-Finetuned/`: the fine-tuned model
  - `Qwen2-0.5B-GRPO-finetuned_OUTPUTS.json`: eval-set generations **after** training

```bash
python grpo_finetune.py
```

> GRPO training needs a CUDA GPU, because the script moves the model to `cuda`.

## Getting Started

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Models download automatically from the Hugging Face Hub on first use. `main.py` calls `load_dotenv()`, so you can put a Hugging Face token (`HF_TOKEN=...`) in a `.env` file if you need one.

## Tech Stack

Python · PyTorch · Hugging Face Transformers & Datasets · TRL (GRPO) · pandas · NumPy

## References

- **CoMAT:** *Chain of Mathematically Annotated Thought Improves Mathematical Reasoning* (Leang et al., 2024)
- **MMLU-Redux:** *Are We Done with MMLU?* (Gema et al., 2024)
- **GRPO:** *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (Shao et al., 2024)
