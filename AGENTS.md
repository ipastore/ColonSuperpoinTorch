
# AGENTS.md

This document tells the Copilot coding agent **how to work in this repository**: 

## 1) Repository layout

- Top repo: **ColonSuperpoinTorch** (this repo).

---

## 2) Environment 

1. Use conda environment
```bash
conda activate py38-sp
```
---

## 3) Code style, typing, docs

* **Style**: Black (line length 88) + isort. Prefer early returns and expressive `snake_case` names.
* **Typing**: Add Python type hints on public functions/classes.
* **Docstrings**: Use **Google-style** docstrings with `Args:` and `Returns:`, focused on **why** as well as what.

Example:

```python
def example(x: int) -> str:
    """Summarize purpose and result.

    Args:
        x: Meaningful description.

    Returns:
        Description of the returned value.
    """
```

---


## 4) Agent task flow (step-by-step)

1. **Plan**

   * Identify the components to change and whether they reside in a leaf submodule.
2. **Prepare**

   * Confirm Python 3.8.
   * activate conda environment
  
3. **Iterate over user entry**
   
   * After giving the first attemp, wait for user input to continue.


