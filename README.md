# QuadConnect2.5-0.5B - A Strategic Connect Four AI

![Connect Four Demo](https://cdn-uploads.huggingface.co/production/uploads/62f847d692950415b63c6011/QiDstnBXlVVz6dGrx3uus.png)

## 🎮 Overview

QuadConnect is a specialized language model trained to master the game of Connect Four. Built on [Qwen 2.5 (0.5B parameter base)](https://huggingface.co/unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit), this model uses GRPO (Group Relative Policy Optimization) to learn the strategic intricacies of Connect Four gameplay.

**Status**: Early training experiments (v0.0.9b) - Reward functions still evolving

## 🔗 Links
- [Model on Hugging Face](https://huggingface.co/Lyte/QuadConnect2.5-0.5B-v0.0.9b)
- [Interactive Demo](https://huggingface.co/spaces/Lyte/QuadConnect-beta)

## 🔍 Model Details

- **Developed by:** [Lyte](https://hf.co/Lyte)
- **Model type:** Small Language Model (SLM)
- **Language:** English
- **Base model:** [unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit](https://huggingface.co/unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit)
- **Training method:** [TRL](https://github.com/huggingface/trl)'s GRPO
- **Training data:** [Lyte/ConnectFour-T10](https://huggingface.co/datasets/Lyte/ConnectFour-T10)

## 🚀 Quick Start

### Option 1: Using Transformers

```python
from transformers import pipeline

SYSTEM_PROMPT = """You are a master Connect Four strategist whose goal is to win while preventing your opponent from winning. The game is played on a 6x7 grid (columns a–g, rows 1–6 with 1 at the bottom) where pieces drop to the lowest available spot.

Board:
- Represented as a list of occupied cells in the format: <column><row>(<piece>), e.g., 'a1(O)'.
- For example: 'a1(O), a2(X), b1(O)' indicates that cell a1 has an O, a2 has an X, and b1 has an O.
- An empty board is shown as 'Empty Board'.
- Win by connecting 4 pieces in any direction (horizontal, vertical, or diagonal).

Strategy:
1. Identify taken positions, and empty positions.
2. Find and execute winning moves.
3. If There isn't a winning move, then block your opponent's potential wins.
4. Control the center and set up future moves.

Respond in XML:
<reasoning>
Explain your thought process, focusing on your winning move, how you block your opponent, and your strategic plans.
</reasoning>
<move>
Specify the column letter (a–g) for your next move.
</move>
"""

board = {
    "empty": "Game State:\n- You are playing as: X\n- Your previous moves: \n- Opponent's moves: \n- Current board state: Empty Board\n- Next available position per column:  \nColumn a: a1, a2, a3, a4, a5, a6  \nColumn b: b1, b2, b3, b4, b5, b6  \nColumn c: c1, c2, c3, c4, c5, c6  \nColumn d: d1, d2, d3, d4, d5, d6  \nColumn e: e1, e2, e3, e4, e5, e6  \nColumn f: f1, f2, f3, f4, f5, f6  \nColumn g: g1, g2, g3, g4, g5, g6\n\nMake your move.",
    "one_move": "Game State:\n- You are playing as: X\n- Your previous moves: \n- Opponent's moves: b1\n- Current board state: b1(O)\n- Next available position per column:  \nColumn a: a1, a2, a3, a4, a5, a6  \nColumn b: b2, b3, b4, b5, b6  \nColumn c: c1, c2, c3, c4, c5, c6  \nColumn d: d1, d2, d3, d4, d5, d6  \nColumn e: e1, e2, e3, e4, e5, e6  \nColumn f: f1, f2, f3, f4, f5, f6  \nColumn g: g1, g2, g3, g4, g5, g6\n\nMake your move.",
    "four_moves": "Game State:\n- You are playing as: X\n- Your previous moves: a1, a2\n- Opponent's moves: d1, a3\n- Current board state: a1(X), d1(O), a2(X), a3(O)\n- Next available position per column:  \nColumn a: a4, a5, a6  \nColumn b: b1, b2, b3, b4, b5, b6  \nColumn c: c1, c2, c3, c4, c5, c6  \nColumn d: d2, d3, d4, d5, d6  \nColumn e: e1, e2, e3, e4, e5, e6  \nColumn f: f1, f2, f3, f4, f5, f6  \nColumn g: g1, g2, g3, g4, g5, g6\n\nMake your move.",
}

generator = pipeline("text-generation", model="Lyte/QuadConnect2.5-0.5B-v0.0.9b", device="cuda")

# use 'empty', 'one_move' or 'four_moves' in board['']
output = generator([
    {"role": "system", "content": SYSTEM_PROMPT}, 
    {"role": "user", "content": board['empty']}
], max_new_tokens=10245, return_full_text=False)[0]

print(output["generated_text"])
```

### Option 2: Using GGUF

Download the [Quantized GGUF (Q8_0)](https://huggingface.co/Lyte/QuadConnect2.5-0.5B-v0.0.9b/blob/main/unsloth.Q8_0.gguf) and use it in your favorite GGUF inference engine (e.g., LMStudio).

### Option 3: Using Hugging Face Space

Visit the [QuadConnect Demo Space](https://huggingface.co/spaces/Lyte/QuadConnect-beta) to interact with the model directly. You can also duplicate the space or download its code for local use.

## 📊 Evaluation Results

Model performance was evaluated on the [Lyte/ConnectFour-T10](https://huggingface.co/datasets/Lyte/ConnectFour-T10) validation split with various temperature settings.

### Summary Metrics Comparison

| Metric | v0.0.6b (Temp 0.6) | v0.0.8b (Temp 0.6) | v0.0.9b (Temp 0.6) | v0.0.9b (Temp 0.8) | v0.0.9b (Temp 1.0) |
|--------|-------------------|-------------------|-------------------|-------------------|-------------------|
| Total games evaluated | 5082 | 5082 | 5082 | 5082 | 5082 |
| Correct predictions | 518 | 394 | 516 | **713** | 677 |
| Accuracy | 10.19% | 7.75% | 10.15% | **14.03%** | 13.32% |
| Most common move | d (41.14%) | d (67.61%) | a (38.72%) | a (31.01%) | a (26.99%) |
| Middle column usage | 75.05% | 99.53% | 29.08% | 35.43% | 39.49% |

## 🔧 Training Details

### Data Preparation
1. Started with [Leon-LLM/Connect-Four-Datasets-Collection](https://huggingface.co/datasets/Leon-LLM/Connect-Four-Datasets-Collection)
2. Filtered for clean, complete entries
3. Further filtered to include only games with 10 or fewer turns
4. Split into train and validation sets
5. Final dataset: [Lyte/ConnectFour-T10](https://huggingface.co/datasets/Lyte/ConnectFour-T10)

### Evaluation Parameters
- Temperature: 0.6, 0.8, 1.0 (compared)
- Top-p: 0.95
- Max tokens: 1024

### Framework Versions
- TRL: 0.15.1
- Transformers: 4.49.0
- PyTorch: 2.5.1+cu121
- Datasets: 3.2.0
- Tokenizers: 0.21.0

## 📚 Citations

For GRPO:
```bibtex
@article{zhihong2024deepseekmath,
    title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
    author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
    year         = 2024,
    eprint       = {arXiv:2402.03300},
}
```

For TRL:
```bibtex
@misc{vonwerra2022trl,
    title        = {{TRL: Transformer Reinforcement Learning}},
    author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallouédec},
    year         = 2020,
    journal      = {GitHub repository},
    publisher    = {GitHub},
    howpublished = {\url{https://github.com/huggingface/trl}}
}
```