
# AGENTS.md

This document tells the AI coding agent **how to work in this repository**: 

## 1) Repository layout

- Top repo: **ColonSuperpoinTorch** (this repo).
---

## 2) Environment 

1. Use conda environment
2. 
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


---

## 4) Branching, rebasing, PRs

* Always create a **feature branch** before making any code or documentation changes. The only exception is clarification-only responses that do not touch the filesystem. Never push directly to `main`.
* Before creating the branch, ensure the worktree is clean. If there are local modifications on `master`, run `git stash push --include-untracked -m "<reason>"` to shelve them, then create the branch from the clean state. Leave the stash untouched while you work so you can decide later—when you return to `master`—whether to reapply it with `git stash pop` or discard it with `git stash drop`.
* **Rebase** your branch onto the latest `main` before merging:

  ```bash
  git fetch origin
  git rebase origin/main
  ```
* **Squash** trivial/iterative commits before merging.
* **PR checklist**:

  * Include a short **What / Why / Test** section.
  * Ensure CI checks (tests) are **green**.
  * For breaking changes, update README/docs.
  * For dependency changes, update packaging/requirements.

After merging, **delete the feature branch**.

---

## 5) Agent task flow (step-by-step)

1. **Plan**

   * Identify the components to change and make a plan in no less than 6 steps.
2. **Prepare**

   * Confirm `conda activate py38-sp` has been run so commands use the correct environment.
   * Run `git status --short` on `master`; if it is not clean, stash the local edits with `git stash push --include-untracked -m "<reason>"` and keep the stash for later review when you return to `master`.
   * When the task involves editing files, create a new branch after confirming the clean state:

     ```bash
     git checkout -b <descriptive-branch-name>
     ```

     Stay on this branch for the remainder of the task.

3. **Implement**

4. **Validate**
   1. For export tasks: 
    ```bash
    python export.py export_detector_homoAdapt configs/superpoint_colon_export_test.yaml ds4_specular_camera_mask_th005_k50_topk600_toy
    ```
   2. For training tasks: 
    ```bash
    python export.py export_detector_homoAdapt configs/superpoint_colon_export_test.yaml ds4_specular_camera_mask_th005_k50_topk600_toy
    ```
For each run, and every task (export or training) the logs are saved under runs/export_detector_homoAdapt (for export tasks) and runs/train_joint.

For each case you can run the tensorboard to analyze the output if necesssary
    
    ```bash
    tensorboard --logdir runs/export_detector_homoAdapt/<name_of_run>
    ```

    ```bash
    tensorboard --logdir runs/train_joint/<name_of_run>
    ```
5. **Hand off for review**

   * Present the changes and wait for user confirmation. Do not commit or merge until the user explicitly requests it.

6. **Prepare PR**

   * Provide **What / Why / Test**, note decisions/assumptions, and keep scope tight (split large changes into smaller PRs).

7. **Safety rails**

   * Do not fetch LFS assets unless the task requires them.

8. **Finalization on request**

   * When the user approves, squash commits into a single commit, rebase onto `master`, and merge as instructed. Delete the feature branch after merging.
