# AI Learning

## Day 1
- Python 基础

## Day 2
- 函数、类、文件读写
- NumPy、Matplotlib

## Day 3
- PyTorch 安装
- MNIST 手写数字识别，CNN
- 测试准确率：98.27%

## 环境
- Conda 环境 `ai`：`C:\Users\tw188\miniconda3\envs\ai\python.exe`
- Python 3.10 / PyTorch 2.14.1+cpu / torchvision 0.29.1+cpu

> 注意：系统 PATH 里的 `python` 是 MSYS2 3.14 或 Windows Store 的占位程序，**都没有 PyTorch**。
> 而且 `conda` 没有加入 PATH，`conda activate ai` 会报“无法识别 conda”。所以直接 `python mnist.py` 跑不了。

## 运行

VS Code 里三种方式任选（`.vscode/` 已配好，指到 `ai` 环境）：

1. **终端**：新开一个终端，直接 `python mnist.py`（`.vscode/settings.json` 已把 `ai` 环境加进终端 PATH）
2. **F5**：选 `Python: mnist.py (ai env)`（`.vscode/launch.json` 里写死了绝对路径）
3. **右上角三角按钮**：需要解释器选中 `ai`。`Ctrl+Shift+P` → `Python: Select Interpreter` → 选 `Python 3.10.22 ('ai': conda)`。之前它用的是 MSYS2 的 Python 3.14，所以报 `No module named 'torch'`

不依赖 VS Code 的兜底命令（任何终端都能用）：

```powershell
& "C:\Users\tw188\miniconda3\envs\ai\python.exe" mnist.py
```

`conda` 没有加入 PATH，直接 `conda activate ai` 会报“无法识别 conda”。要修的话执行一次：

```powershell
& "C:\Users\tw188\miniconda3\condabin\conda.bat" init powershell
```

MNIST 数据集放在 `./data/MNIST/raw/`，代码里 `download=False` 离线加载。


## 实验记录 1：batch_size 的影响
- **基线**：batch_size=64, epochs=1, Test Accuracy=98.27%
- **实验**：batch_size=128, epochs=1, Test Accuracy=98.09%, Loss=0.0237, 耗时=14.33s
- **分析**：batch_size 增大导致单轮迭代次数减半（938 -> 469），参数更新次数减少，单轮 1 epoch 下准确率微降。需增加 epochs 以充分收敛。


## 实验记录 1：batch_size 的影响
- **基线**：batch_size=64, epochs=1, Test Accuracy=98.27%
- **实验 A**：batch_size=128, epochs=1, Test Accuracy=98.09%, Loss=0.0237, 耗时=14.33s
  - 分析：batch_size 增大导致单轮迭代次数减半，参数更新减少，单轮 1 epoch 下准确率微降。
- **实验 B**：batch_size=128, epochs=3, Test Accuracy=98.96%, 耗时=35.92s
  - 分析：通过增加 epoch 弥补更新次数，模型充分收敛，准确率提升。