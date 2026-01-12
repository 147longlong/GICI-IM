# GICI-IM 项目 Python 桥接修复与环境配置文档

本文档总结了解决 C++ 调用 FastSAM/ONNX 图像分割时遇到的库版本冲突及崩溃问题的修复方案，并提供了编译、运行及环境迁移的指南。

## 1. 问题回顾

非常经典且棘手的**“环境隔离与ABI（二进制接口）兼容性”**问题，特别是在 Linux C++ 程序中嵌入 Conda Python 环境时极易遇到。

操作系统自带的库（System Libs）与 Conda 环境中的库（Conda Libs）发生了版本冲突和路径混淆。


### A.  CMakeLists.txt 中的问题
1.  **强制指定链接libstdc++.so.6**: ABI冲突，ultralytics（以及它依赖的 PyTorch、NumPy 等）是 Conda 里的 Python 包，它们在编译时使用较新的 GCC 版本（例如 GCC 12 或更高），因此依赖较新版本的 C++ 标准库 (libstdc++.so.6)，需要其中的符号（如 GLIBCXX_3.4.30）。但Ubuntu20.04 自带的 libstdc++.so.6 版本较旧，不包含这些符号，导致运行时找不到对应版本。因此需要强制链接Conda环境中的新版 libstdc++.so.6。

2.  **Python 版本冲突**: 系统 Python 版本为 3.8，而 Conda 环境中使用的是 Python 3.10。C++ 程序如果链接了系统的 Python 库， 会导致找不到对应的符号和模块。

3.  **指定opencv和yaml库路径**: 当你安装 Conda 时，Conda 环境里往往也包含了一套 libopencv、libyaml、libjpeg 等库。你的 C++ 代码（GICI-IM）原本是基于系统安装的 OpenCV（在 local 下）开发的。当你运行 cmake 时，CMake 的 find_package 可能会在你的 Conda 路径中先找到 Conda 版的 OpenCV。Conda 版的 OpenCV 通常是精简版（headless）或者不含某些 header，且与系统库不兼容。你需要显式告诉 CMake：“不要去 Conda 里找，去我指定的系统路径找”。


### B.  export LD_LIBRARY_PATH

 Linux 动态链接器（Dynamic Linker/Loader）问题。编译时 (Build time)：你在 CMake 里写了绝对路径（如第 84 行），链接器（ld）知道去哪里找库来验证符号。运行时 (Run time)：当你运行程序 ./gici_main 时，Linux 动态加载器（ld-linux.so）默认会先去 lib 和 lib 找动态库。它会在 lib 找到那个老版本的 libstdc++.so.6 并加载。等你程序运行到一半，试图加载 Conda 的 Python 扩展时，发现内存里已经有一个老版 libstdc++ 了，新版扩展无法使用，导致崩溃或符号错误。通过环境变量 LD_LIBRARY_PATH，你告诉 Linux 加载器：“在去系统目录找之前，先去 Conda 的目录（envs/gici/lib）找”。这样就加载了新版的 C++ 库，向下兼容了系统库，同时满足了 Python 库的需求。

### C. 降级Conda也没用

只要你依然需要在 C++ 代码中同时使用 “操作系统的原生库” (OpenCV/System) 和 “最新 AI 库” (Conda/Ultralytics)，这种冲突就是物理存在的。ultralytics 依赖 PyTorch，而 PyTorch 为了性能，通常是用较新的 GCC 编译器构建的。这意味着无论你怎么降级，只要是用 Conda/Pip 下下来的现代 AI 库，它们几乎都要求比 Ubuntu 20.04/18.04 自带版本更新的 C++ 标准库。除非你手动从源码编译 PyTorch 和 Ultralytics，且编译时强制使用你系统自带的老旧 GCC。但这比修改 CMakeLists 痛苦一百倍。
---


## 2. 环境迁移指南 (Migration Guide)

在新设备上部署时，除了源码和模型文件，需要重建一致的 Conda 环境 (`gici`)。

### 关键依赖包清单

1.  **Python 版本**: 3.10
2.  **Conda 渠道包**:
    *   `pillow` (必须用 conda 安装)
    *   `openjpeg` (必须从 **conda-forge** 安装，版本建议 2.5.x)
3.  **Pip 包**:
    *   `ultralytics`
    *   `numpy<2` (**非常重要**：必须锁定在 1.x 版本，例如 1.26.4，否则会报错)
    *   `opencv-python-headless` (**非常重要**：**禁止**安装 `opencv-python`，必须用 headless 版本以避免 GUI 库冲突)

### 环境创建参考命令

```bash
# 1. 创建基础环境
conda create -n gici python=3.10 -y
conda activate gici

# 2. 安装 Conda 侧依赖 (解决底层库冲突)
conda install -c conda-forge openjpeg -y
conda install pillow -y

# 3. 安装 Pip 侧依赖 (算法库)
# 注意：限制 numpy 版本，并使用 headless opencv
pip install "numpy<2.0.0"
pip install "opencv-python-headless<4.11"
pip install ultralytics pandas seaborn
```
