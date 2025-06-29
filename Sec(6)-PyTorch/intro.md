# 📌 PyTorch - Overview, Features & Comparison

## 📚 What is PyTorch?

**PyTorch** is an **open-source deep learning library** developed by **Facebook's AI Research Lab (FAIR)**.
It is widely used for **research and production** in machine learning and deep learning tasks.

---

## ⭐ Key Features of PyTorch:

* **Dynamic Computational Graphs (Define-by-Run):**

  * Graphs are built on the fly during runtime.
  * Easier debugging and more intuitive Pythonic coding.

* **Tensor Computation (like NumPy, but with GPU acceleration):**

  * Supports CUDA for running on NVIDIA GPUs.

* **Rich Ecosystem:**

  * Libraries like **TorchVision**, **TorchText**, **TorchAudio**, etc.

* **Strong Community Support:**

  * Popular in both research and industry.

* **Autograd Module:**

  * Automatic differentiation for neural networks and optimization.

* **Interoperable with NumPy:**

  * Easy conversion between NumPy arrays and PyTorch tensors.

---

## 📝 How to Install PyTorch:

### For non local usage refer to [link](https://github.com/Aman071106/MLOPS/tree/main/2.Notebooks%2Ceditors)

### Using pip:(CPU)

```bash
pip install torch torchvision torchaudio
```

### For specific CUDA versions (Example: CUDA 11.8):(for gpu usage)
* Check your CUDA version in cmd by typing:
```bash
nvidia-smi
```

* Visit: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## 💡 Code Snippet: Tensor Creation in PyTorch vs TensorFlow

### PyTorch:

```python
import torch
x = torch.tensor([[1, 2], [3, 4]])
print("PyTorch Tensor:")
print(x)
print("Device:", x.device)
```

### TensorFlow:

```python
import tensorflow as tf
x = tf.constant([[1, 2], [3, 4]])
print("TensorFlow Tensor:")
print(x)
print("Device:", x.device)
```

### ✅ Key Difference:

* **PyTorch:** Eager execution by default (dynamic graph).
* **TensorFlow:** Originally static graph (needs tf.function or sessions in older TF versions), but newer TF 2.x versions also support eager execution.

---

## 🔗 Useful Links:

* [PyTorch Official Site](https://pytorch.org/)
* [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

*Happy Learning Deep Learning!*
