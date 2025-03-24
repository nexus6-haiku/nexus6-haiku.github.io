---
title: "Setting up an AI/LLM Stack in Haiku: A Practical Guide part I"
date: 2025-03-19T18:30:00-00:00
categories:
  - New Adventures In AI
tags:
  - AI
  - Haiku
  - Python
---

I wanted to push the boundaries of what Haiku can do, so I decided to experiment with setting up a complete AI stack on it. My goal was to see if Haiku could actually run modern Large Language Models without GPU acceleration by using the most used frameworks like LangChain. While mainstream operating systems often require powerful hardware for AI workloads, I was curious if Haiku might offer a practical alternative for enthusiasts who want to explore AI without investing in specialized equipment. In this article, I'll walk you through how I built a functional Python environment for AI in Haiku and demonstrate how to leverage essential components for working with LLMs, all running on modest hardware like my ThinkPad T480s.

## Preparing the Python Environment

Before we can use AI and machine learning tools, we need to properly configure the Python environment in Haiku. Here are the necessary steps:

### Installing System Packages via Haiku Depot

Let's start by installing the necessary system packages. These packages will provide the foundation on which we'll build our AI stack:

```bash
pkgman install numpy_python310 psutil_python310 pymupdf_python310 pyzmq_python310 setuptools_python310 faiss_python310

```

These packages include:

| Package | Version |
| --- | --- |
| numpy | 2.2.3 |
| psutil | 6.0.0 |
| pymupdf | 1.20.2 |
| pyzmq | 25.1.2 |
| setuptools | 68.2.2 |

### Creating the Virtual Environment

Virtual environments allow us to isolate our project dependencies, avoiding conflicts between packages:

```bash
mkdir project
cd project
python3 -m venv ai-env
source ai-env/bin/activate
python3 -m pip install --upgrade pip

```

It's important to note that Pip will remain installed at the system level. We can verify this with:

```bash
which pip

```

This should show `/bin/pip`

The Python interpreter, however, should point to the virtual environment:

```bash
which python3
```

This should return something like `/Dati/workspace/Python/ai-env/myenv/bin/python3`

### Verifying Installed Packages

It's essential to always use `python3 -m pip` to install packages, otherwise they'll be installed at the system level, regardless of the active environment:

```bash
python3 -m pip list
```

If system-level packages aren't detected, you should only see:

| Package | Version |
| --- | --- |
| pip | 25.0.1 |
| setuptools | 68.2.2 |

Check if `include-system-site-packages = true` is set in the `./ai-env/pyvenv.cfg` file, in which case you should see the complete list of packages.

### Configuring the PATH

Let's add the local path for non-packaged binaries:

```bash
export PATH="/Dati/workspace/Python/ai-env/myenv/non-packaged/bin:$PATH"
echo $PATH
```

### Installing Required Packages

Now we can install the specific packages for AI/LLM:

```bash
python3 -m pip install --no-cache-dir openai langchain langchain-community langchain-openai
```

We use `--no-cache-dir` to ensure that pip doesn't pick up packages from a previous installation.

## Installing Specific Components

### Llama.cpp

Llama.cpp is an efficient LLM implementation that can work even on CPUs:

```bash
pkgman install llama_cpp
export LLAMA_CPP_LIB=/boot/system/lib/libllama.so
export LLAMA_CPP_LIB_PATH=/boot/system/lib/
CMAKE_ARGS="-DLLAMA_BUILD=OFF" python3 -m pip install --no-cache-dir llama-cpp-python
```

### FAISS

FAISS (Facebook AI Similarity Search) has already been installed via pkgman in the previous steps.

### Jupyter

For a more interactive development experience, let's install Jupyter:

```bash
python3 -m pip install --no-cache-dir notebook
cd /Dati/workspace/Python
python3 -m venv jupyter-env
source jupyter-env/bin/activate
export PATH="/Dati/workspace/Python/jupyter-env/non-packaged/bin:$PATH"
export JUPYTERLAB_DIR=/Dati/workspace/Python/jupyter-env/non-packaged/share/jupyter/lab
jupyter lab --ip 127.0.0.1 --port 8888 --allow-root --no-browser
```

Now that we've configured the environment, in the [next post]({% post_url 2025-19-03-Setup-an-environment-for-AI-in-Haiku-Part-2 %}) we will see how to use these tools with some practical examples.