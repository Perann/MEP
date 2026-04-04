# Project realized for the Mise en prodution class

## Project description

---

## 🛠 How to Contribute

This project uses **[Pixi](https://pixi.sh)** as a modern package and workflow manager. It provides a reproducible, multi-platform environment (Linux, macOS, Windows) and handles both Python and system-level dependencies (like CUDA or C++ libraries) without interfering with your global system.

---

### 1. Prerequisites
You do **not** need to install Python or Conda manually. Simply install the Pixi CLI:

```bash
# On macOS or Linux:
curl -fsSL [https://pixi.sh/install.sh](https://pixi.sh/install.sh) | bash
```

### 2. Environment setup

```bash
git clone <your-repo-url>
cd <your-project-name>
pixi install
```

### 3. Working with the Environment

There are two ways to interact with the project:

Task-based (Recommended): Run commands directly via Pixi. This ensures the environment is always correctly loaded.

```bash
pixi run python src/train.py
```

Interactive Shell: Activate the environment in your current terminal (similar to conda activate):

```bash
pixi shell
```

(Type exit to leave the shell)

### 4. Code Quality (Pre-commit)

We use Black to maintain a consistent coding style. You must install the git hooks before making your first commit : 

```bash
pixi run pre-commit install
```
