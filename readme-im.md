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

# 4. 运行配置 (重要：不要在 .bashrc 中全局 export LD_LIBRARY_PATH)
# 全局设置会导致系统命令(cmake/make)崩溃。请使用以下两种方法之一：

# 方法 A：使用 Alias (推荐，写入 .bashrc)
echo 'alias run_gici="LD_LIBRARY_PATH=/home/syl/miniconda3/envs/gici/lib:\$LD_LIBRARY_PATH ./gici_main"' >> ~/.bashrc
source ~/.bashrc
# 使用方式：直接输入 run_gici ../option/xxx

# 方法 B：每次运行手动指定
LD_LIBRARY_PATH="/home/syl/miniconda3/envs/gici/lib:$LD_LIBRARY_PATH" 
./gici_main ../option/pseudo_real_time_estimation_RTK_RRR.yaml
```

!!注意，这个必须指定，否则主程序就会在运行时候找不到正确的 libstdc++.so.6，出现下面问题：
```
./gici_main: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/syl/GICI-IM/build/libgici.so)
```

## 3. 项目结构速查表（给 AI 的上下文压缩）

目标：用“目录 → 文件 → 一句话用途”的方式，快速建立对仓库的心智模型；对于体量巨大或自动生成目录（如 `build/`、`ros_wrapper/build/`、`results/` 的图片/日志），做聚类摘要而不逐个展开。以下完整性、完好性、integrity均指一个意思。

### 3.1 顶层入口（Top-level）

- `CMakeLists.txt`：顶层 CMake 构建脚本（依赖查找、Conda/libstdc++ 链接策略、生成 `libgici.so` 与可执行）。
- `README.md`：项目主说明（依赖、编译、运行、ROS wrapper 使用等）。
- `readme-im.md`：本地环境/桥接问题记录 + 本章节“结构速查”。
- `LICENSE`：许可证。
- `.git/`：Git 元数据。
- `.gitignore`：Git 忽略规则。
- `.vscode/launch.json`：VS Code 调试配置。
- `.vscode/settings.json`：VS Code 工作区设置。
- `build/`：CMake 构建产物目录（缓存、Makefile、可执行、`compile_commands.json` 等；可删后重建）。
- `doc/`：文档目录。
- `doc/manual.pdf`：项目/算法使用手册（PDF）。
- `include/`：对外头文件（库 API，和 `src/` 一一对应）。
- `src/`：核心实现（GNSS/IMU/Vision/Fusion/Integrity/Stream/Utility）。
- `option/`：YAML 配置与辅助数据（驱动不同模式运行）。
- `tools/`：离线工具/格式转换/评估脚本与小型可执行。
- `third_party/`：第三方依赖源码与模型资产（FAST/SVO/VIKIT/RTKLIB/Segmentation）。
- `ros_wrapper/`：ROS(catkin) 封装包（含消息定义、ROS 主入口、发布/订阅接口）。
- `integrity_test/`：完整性算法/统计相关的独立测试与小程序。
- `results/`：实验输出、可视化脚本、日志与图表（通常为运行后生成/手工分析产物）。
- `visual_ism/`：视觉完整性/ISM 相关的离线分析脚本与图表/参数文件。

### 3.2 可执行入口（Runtime Entry Points）

- `src/gici_main.cpp`：主程序入口（读取 YAML，创建/运行 `NodeHandle` + `SpinControl`）。
- `src/integrity_main.cpp`：完整性后处理入口（读取 integrity YAML、读取快照、生成/更新 NMEA/CSV/统计结果）。

### 3.3 核心源码（`src/`）

#### `src/estimate/`（优化/图优化基础）

- `src/estimate/ceres_iteration_callback.cpp`：Ceres 迭代回调实现。
- `src/estimate/error_interface.cpp`：误差项接口实现。
- `src/estimate/estimating.cpp`：估计流程实现。
- `src/estimate/estimator_base.cpp`：估计器基类实现。
- `src/estimate/estimator_types.cpp`：估计相关类型实现。
- `src/estimate/graph.cpp`：因子图/图结构实现。
- `src/estimate/homogeneous_point_local_parameterization.cpp`：齐次点局部参数化实现。
- `src/estimate/homogeneous_point_parameter_block.cpp`：齐次点参数块实现。
- `src/estimate/local_parameterization_additional_interfaces.cpp`：参数化扩展接口实现。
- `src/estimate/marginalization_error.cpp`：边缘化误差实现。
- `src/estimate/motion_detector.cpp`：运动检测实现。
- `src/estimate/pose_error.cpp`：位姿误差实现。
- `src/estimate/pose_local_parameterization.cpp`：位姿局部参数化实现。
- `src/estimate/pose_parameter_block.cpp`：位姿参数块实现。
- `src/estimate/speed_and_bias_parameter_block.cpp`：速度/偏置参数块实现。

#### `src/fusion/`（多传感器融合估计器）

- `src/fusion/gnss_imu_camera_srr_estimator.cpp`：GNSS+IMU+Camera（SRR）融合估计器实现。
- `src/fusion/gnss_imu_initializer.cpp`：GNSS-IMU 初始化实现。
- `src/fusion/gnss_imu_lc_estimator.cpp`：GNSS+IMU（LC）融合估计器实现。
- `src/fusion/multisensor_estimating.cpp`：多传感器估计调度/封装实现。
- `src/fusion/ppp_imu_tc_estimator.cpp`：PPP+IMU（TC）融合估计器实现。
- `src/fusion/rtk_imu_camera_rrr_estimator.cpp`：RTK+IMU+Camera（RRR）融合估计器实现。
- `src/fusion/rtk_imu_tc_estimator.cpp`：RTK+IMU（TC）融合估计器实现。
- `src/fusion/spp_imu_camera_rrr_estimator.cpp`：SPP+IMU+Camera（RRR）融合估计器实现。
- `src/fusion/spp_imu_tc_estimator.cpp`：SPP+IMU（TC）融合估计器实现。

#### `src/gnss/`（GNSS 观测/误差/解算）

- `src/gnss/ambiguity_common.cpp`：载波模糊度公共逻辑实现。
- `src/gnss/ambiguity_error.cpp`：模糊度误差项实现。
- `src/gnss/ambiguity_resolution.cpp`：模糊度固定/解算实现。
- `src/gnss/ambiguity_resolution_differential.cpp`：差分场景模糊度解算实现。
- `src/gnss/code_bias.cpp`：码偏差相关实现。
- `src/gnss/dgnss_estimator.cpp`：DGNSS 估计器实现。
- `src/gnss/doppler_error.cpp`：多普勒误差项实现。
- `src/gnss/geodetic_coordinate.cpp`：大地坐标/坐标变换实现。
- `src/gnss/gnss_common.cpp`：GNSS 公共工具实现。
- `src/gnss/gnss_estimator_base.cpp`：GNSS 估计器基类实现。
- `src/gnss/gnss_estimator_base_differential.cpp`：差分 GNSS 估计器基类实现。
- `src/gnss/gnss_estimator_base_logger.cpp`：GNSS 估计器日志/输出实现。
- `src/gnss/gnss_loose_estimator_base.cpp`：松耦合 GNSS 估计器基类实现。
- `src/gnss/gnss_types.cpp`：GNSS 类型实现。
- `src/gnss/phase_bias.cpp`：相位偏差相关实现。
- `src/gnss/phaserange_error.cpp`：载波相位/相距误差实现。
- `src/gnss/phaserange_error_dd.cpp`：双差载波相位误差实现。
- `src/gnss/phaserange_error_sd.cpp`：单差载波相位误差实现。
- `src/gnss/phase_windup.cpp`：相位缠绕（wind-up）实现。
- `src/gnss/position_error.cpp`：位置误差项实现。
- `src/gnss/ppp_estimator.cpp`：PPP 估计器实现。
- `src/gnss/pseudorange_error.cpp`：伪距误差实现。
- `src/gnss/pseudorange_error_dd.cpp`：双差伪距误差实现。
- `src/gnss/pseudorange_error_sd.cpp`：单差伪距误差实现。
- `src/gnss/relative_isb_error.cpp`：相对 ISB 误差实现。
- `src/gnss/rtk_estimator.cpp`：RTK 估计器实现。
- `src/gnss/sdgnss_estimator.cpp`：SDGNSS 估计器实现。
- `src/gnss/spp_estimator.cpp`：SPP 估计器实现。
- `src/gnss/velocity_error.cpp`：速度误差项实现。

#### `src/imu/`（IMU 误差/约束/估计）

- `src/imu/hmc_error.cpp`：HMC 相关误差项实现。
- `src/imu/imu_common.cpp`：IMU 公共工具实现。
- `src/imu/imu_error.cpp`：IMU 误差项实现。
- `src/imu/imu_estimator_base.cpp`：IMU 估计器基类实现。
- `src/imu/nhc_error.cpp`：NHC（非完整约束）误差实现。
- `src/imu/roll_and_pitch_error.cpp`：横滚/俯仰误差实现。
- `src/imu/speed_and_bias_error.cpp`：速度与偏置误差实现。
- `src/imu/yaw_error.cpp`：航向误差实现。

#### `src/integrity/`（完整性/可视化/ISM）

- `src/integrity/jacobian_visualization.cpp`：雅可比结构/可视化相关实现。
- `src/integrity/visual_integrity.cpp`：视觉完整性处理（快照保存/读取、后处理、统计）实现。
- `src/integrity/visual_ism_gen.cpp`：视觉 ISM 相关生成/处理实现。

#### `src/stream/`（数据流：streamer/formator/integration）

- `src/stream/data_integration.cpp`：多源数据集成/对齐实现。
- `src/stream/format_image.c`：图像格式化（C 实现）。
- `src/stream/format_imu.c`：IMU 格式化（C 实现）。
- `src/stream/formator.cpp`：数据格式化器实现。
- `src/stream/node_handle.cpp`：节点句柄（模块装配/生命周期）实现。
- `src/stream/streamer.cpp`：数据输入流（文件/网络/设备）实现。
- `src/stream/streaming.cpp`：streaming 调度与运行实现。

#### `src/utility/`（通用工具与配置解析）

- `src/utility/common.cpp`：通用函数实现。
- `src/utility/global_variable.cpp`：全局变量/配置实现。
- `src/utility/node_option_handle.cpp`：YAML/option 解析与装配实现。
- `src/utility/option.cpp`：Option 数据结构实现。
- `src/utility/signal_handle.cpp`：信号处理（退出/中断）实现。
- `src/utility/spin_control.cpp`：主循环/线程调度控制实现。
- `src/utility/transform.cpp`：坐标/变换工具实现。

#### `src/vision/`（视觉前端/误差项/分割桥接）

- `src/vision/feature_handler.cpp`：特征管理实现。
- `src/vision/feature_matcher.cpp`：特征匹配实现。
- `src/vision/feature_tracker.cpp`：特征跟踪实现。
- `src/vision/homogeneous_point_error.cpp`：齐次点视觉误差实现。
- `src/vision/python_segmentation_bridge.cpp`：Python 分割（FastSAM/ONNX）桥接实现。
- `src/vision/relative_pose_error.cpp`：相对位姿误差实现。
- `src/vision/segmentator.cpp`：分割器封装实现。
- `src/vision/visual_estimator_base.cpp`：视觉估计器基类实现。
- `src/vision/visual_initialization.cpp`：视觉初始化实现。

### 3.4 头文件 API（`include/`）

说明：`include/gici/**` 与 `src/**` 基本一一对应；一般先看头文件理解接口/Option，再看对应实现。

#### `include/gici/estimate/`

- `include/gici/estimate/ceres_iteration_callback.h`：Ceres 迭代回调接口。
- `include/gici/estimate/common_parameter_block.h`：通用参数块定义。
- `include/gici/estimate/const_error.h`：常量约束误差定义。
- `include/gici/estimate/error_interface.h`：误差项接口定义。
- `include/gici/estimate/estimating.h`：估计流程接口。
- `include/gici/estimate/estimator_base.h`：估计器基类接口。
- `include/gici/estimate/estimator_types.h`：估计相关类型定义。
- `include/gici/estimate/graph.h`：图/因子图结构定义。
- `include/gici/estimate/homogeneous_point_local_parameterization.h`：齐次点局部参数化定义。
- `include/gici/estimate/homogeneous_point_parameter_block.h`：齐次点参数块定义。
- `include/gici/estimate/local_parameterization_additional_interfaces.h`：参数化扩展接口定义。
- `include/gici/estimate/marginalization_error.h`：边缘化误差定义。
- `include/gici/estimate/marginalization_error_impl.h`：边缘化误差实现细节。
- `include/gici/estimate/motion_detector.h`：运动检测接口。
- `include/gici/estimate/parameter_block.h`：参数块基类定义。
- `include/gici/estimate/pose_error.h`：位姿误差定义。
- `include/gici/estimate/pose_local_parameterization.h`：位姿局部参数化定义。
- `include/gici/estimate/pose_parameter_block.h`：位姿参数块定义。
- `include/gici/estimate/relative_const_error.h`：相对常量约束误差定义。
- `include/gici/estimate/relative_integration_error.h`：相对积分误差定义。
- `include/gici/estimate/speed_and_bias_parameter_block.h`：速度/偏置参数块定义。

#### `include/gici/fusion/`

- `include/gici/fusion/gnss_imu_camera_srr_estimator.h`：GNSS+IMU+Camera SRR 估计器接口。
- `include/gici/fusion/gnss_imu_initializer.h`：GNSS-IMU 初始化接口。
- `include/gici/fusion/gnss_imu_lc_estimator.h`：GNSS+IMU LC 估计器接口。
- `include/gici/fusion/multisensor_estimating.h`：多传感器估计调度接口。
- `include/gici/fusion/multisensor_initializer_base.h`：多传感器初始化基类接口。
- `include/gici/fusion/ppp_imu_tc_estimator.h`：PPP+IMU TC 估计器接口。
- `include/gici/fusion/rtk_imu_camera_rrr_estimator.h`：RTK+IMU+Camera RRR 估计器接口。
- `include/gici/fusion/rtk_imu_tc_estimator.h`：RTK+IMU TC 估计器接口。
- `include/gici/fusion/spp_imu_camera_rrr_estimator.h`：SPP+IMU+Camera RRR 估计器接口。
- `include/gici/fusion/spp_imu_tc_estimator.h`：SPP+IMU TC 估计器接口。

#### `include/gici/gnss/`

- `include/gici/gnss/ambiguity_common.h`：模糊度公共定义。
- `include/gici/gnss/ambiguity_error.h`：模糊度误差定义。
- `include/gici/gnss/ambiguity_resolution.h`：模糊度解算接口。
- `include/gici/gnss/code_bias.h`：码偏差定义。
- `include/gici/gnss/code_phase_maps.h`：码/相位相关映射定义。
- `include/gici/gnss/dgnss_estimator.h`：DGNSS 估计器接口。
- `include/gici/gnss/differential_measurement_align.h`：差分观测对齐定义。
- `include/gici/gnss/doppler_error.h`：多普勒误差定义。
- `include/gici/gnss/geodetic_coordinate.h`：坐标/大地坐标定义。
- `include/gici/gnss/gnss_common.h`：GNSS 公共工具定义。
- `include/gici/gnss/gnss_const_errors.h`：GNSS 常量约束误差定义。
- `include/gici/gnss/gnss_estimator_base.h`：GNSS 估计器基类接口。
- `include/gici/gnss/gnss_loose_estimator_base.h`：松耦合 GNSS 估计器基类接口。
- `include/gici/gnss/gnss_parameter_blocks.h`：GNSS 参数块定义。
- `include/gici/gnss/gnss_relative_errors.h`：GNSS 相对误差定义。
- `include/gici/gnss/gnss_types.h`：GNSS 类型定义。
- `include/gici/gnss/phase_bias.h`：相位偏差定义。
- `include/gici/gnss/phase_center.h`：天线相位中心相关定义。
- `include/gici/gnss/phaserange_error_dd.h`：双差相位误差定义。
- `include/gici/gnss/phaserange_error.h`：相位/相距误差定义。
- `include/gici/gnss/phaserange_error_sd.h`：单差相位误差定义。
- `include/gici/gnss/phase_windup.h`：相位缠绕定义。
- `include/gici/gnss/position_error.h`：位置误差定义。
- `include/gici/gnss/ppp_estimator.h`：PPP 估计器接口。
- `include/gici/gnss/pseudorange_error_dd.h`：双差伪距误差定义。
- `include/gici/gnss/pseudorange_error.h`：伪距误差定义。
- `include/gici/gnss/pseudorange_error_sd.h`：单差伪距误差定义。
- `include/gici/gnss/relative_isb_error.h`：相对 ISB 误差定义。
- `include/gici/gnss/rtk_estimator.h`：RTK 估计器接口。
- `include/gici/gnss/sdgnss_estimator.h`：SDGNSS 估计器接口。
- `include/gici/gnss/spp_estimator.h`：SPP 估计器接口。
- `include/gici/gnss/velocity_error.h`：速度误差定义。

#### `include/gici/imu/`

- `include/gici/imu/hmc_error.h`：HMC 误差定义。
- `include/gici/imu/imu_common.h`：IMU 公共工具定义。
- `include/gici/imu/imu_error.h`：IMU 误差定义。
- `include/gici/imu/imu_estimator_base.h`：IMU 估计器基类接口。
- `include/gici/imu/imu_types.h`：IMU 类型定义。
- `include/gici/imu/nhc_error.h`：NHC 误差定义。
- `include/gici/imu/roll_and_pitch_error.h`：横滚/俯仰误差定义。
- `include/gici/imu/speed_and_bias_error.h`：速度/偏置误差定义。
- `include/gici/imu/yaw_error.h`：航向误差定义。

#### `include/gici/integrity/`

- `include/gici/integrity/jacobian_visualization.h`：雅可比可视化接口。
- `include/gici/integrity/visual_integrity.h`：视觉完整性模块接口与选项定义。
- `include/gici/integrity/visual_ism_gen.h`：视觉 ISM 生成/处理接口。

#### `include/gici/stream/`

- `include/gici/stream/data_integration.h`：数据集成接口。
- `include/gici/stream/format_image.h`：图像格式化接口。
- `include/gici/stream/format_imu.h`：IMU 格式化接口。
- `include/gici/stream/formator.h`：格式化器接口。
- `include/gici/stream/node_handle.h`：NodeHandle 接口。
- `include/gici/stream/streamer.h`：Streamer 接口。
- `include/gici/stream/streaming.h`：Streaming 调度接口。

#### `include/gici/utility/`

- `include/gici/utility/common.h`：通用工具接口。
- `include/gici/utility/global_variable.h`：全局变量接口。
- `include/gici/utility/node_option_handle.h`：YAML/Option 解析接口。
- `include/gici/utility/option.h`：Option 数据结构定义。
- `include/gici/utility/rtklib_safe.h`：RTKLIB 安全封装/兼容接口。
- `include/gici/utility/signal_handle.h`：信号处理接口。
- `include/gici/utility/spin_control.h`：SpinControl 接口。
- `include/gici/utility/svo.h`：SVO 相关桥接/封装接口。
- `include/gici/utility/transform.h`：变换工具接口。

#### `include/gici/vision/`

- `include/gici/vision/epipolar_error.h`：对极约束误差定义。
- `include/gici/vision/feature_handler.h`：特征管理接口。
- `include/gici/vision/feature_matcher.h`：特征匹配接口。
- `include/gici/vision/feature_tracker.h`：特征跟踪接口。
- `include/gici/vision/homogeneous_point_error.h`：齐次点视觉误差定义。
- `include/gici/vision/image_types.h`：图像相关类型定义。
- `include/gici/vision/python_segmentation_bridge.h`：Python 分割桥接接口。
- `include/gici/vision/relative_pose_error.h`：相对位姿误差定义。
- `include/gici/vision/reprojection_error_base.h`：重投影误差基类。
- `include/gici/vision/reprojection_error.h`：重投影误差定义。
- `include/gici/vision/reprojection_error_impl.h`：重投影误差实现细节。
- `include/gici/vision/reprojection_error_simple.h`：简化重投影误差。
- `include/gici/vision/segmentator.h`：分割器接口。
- `include/gici/vision/visual_estimator_base.h`：视觉估计器基类接口。
- `include/gici/vision/visual_initialization.h`：视觉初始化接口。

### 3.5 配置与数据（`option/`）

- `option/data_broadcast.yaml`：数据广播/输出相关配置示例。
- `option/data_storage.yaml`：数据存储相关配置示例。
- `option/format_conversion_and_broadcast.yaml`：格式转换 + 广播示例配置。
- `option/format_conversion_and_storage.yaml`：格式转换 + 存储示例配置。
- `option/real_time_estimation.yaml`：实时估计示例配置。
- `option/pseudo_real_time_estimation_DGNSS.yaml`：DGNSS 伪实时估计配置。
- `option/pseudo_real_time_estimation_LC.yaml`：LC 伪实时估计配置。
- `option/pseudo_real_time_estimation_PPP_TC.yaml`：PPP-TC 伪实时估计配置。
- `option/pseudo_real_time_estimation_PPP.yaml`：PPP 伪实时估计配置。
- `option/pseudo_real_time_estimation_RTK_RRR.yaml`：RTK-RRR 伪实时估计配置。
- `option/pseudo_real_time_estimation_RTK_TC.yaml`：RTK-TC 伪实时估计配置。
- `option/pseudo_real_time_estimation_RTK.yaml`：RTK 伪实时估计配置。
- `option/pseudo_real_time_estimation_SDGNSS.yaml`：SDGNSS 伪实时估计配置。
- `option/pseudo_real_time_estimation_SPP_RRR.yaml`：SPP-RRR 伪实时估计配置。
- `option/pseudo_real_time_estimation_SPP_TC.yaml`：SPP-TC 伪实时估计配置。
- `option/pseudo_real_time_estimation_SPP.yaml`：SPP 伪实时估计配置。
- `option/pseudo_real_time_estimation_SRR.yaml`：SRR 伪实时估计配置（本次对话涉及 integrity_options）。
- `option/igs14.atx`：天线相位中心（ANTEX）文件。
- `option/CAS0MGXRAP_20221580000_01D_01D_DCB.BSX`：DCB（码偏差）数据文件。
- `option/gici-mask.png`：mask 示例图片。

### 3.6 工具与脚本（`tools/`）

#### `tools/conversions/`（C 小工具）

- `tools/conversions/coordinate_converter/CMakeLists.txt`：坐标转换工具构建脚本。
- `tools/conversions/coordinate_converter/src/deg_to_dms.c`：角度格式转换。
- `tools/conversions/coordinate_converter/src/ecef_to_lla.c`：ECEF → LLA。
- `tools/conversions/coordinate_converter/src/enu_to_lla.c`：ENU → LLA。
- `tools/conversions/coordinate_converter/src/lla_to_ecef.c`：LLA → ECEF。
- `tools/conversions/coordinate_converter/src/lla_to_enu.c`：LLA → ENU。
- `tools/conversions/time_converter/CMakeLists.txt`：时间转换工具构建脚本。
- `tools/conversions/time_converter/src/gpst_to_unix.c`：GPST → Unix。
- `tools/conversions/time_converter/src/unix_to_epoch.c`：Unix → Epoch。
- `tools/conversions/time_converter/src/unix_to_gpst.c`：Unix → GPST。

#### `tools/edit_binary/`（二进制数据编辑）

- `tools/edit_binary/edit_timestamp/`：时间戳校验/修正/裁剪工具。
    - `tools/edit_binary/edit_timestamp/include/edit_timestamp_utility.h`：工具函数声明。
    - `tools/edit_binary/edit_timestamp/option/cut_gic_files.yaml`：裁剪示例配置。
    - `tools/edit_binary/edit_timestamp/src/check_image_timestamp.cpp`：检查图像时间戳。
    - `tools/edit_binary/edit_timestamp/src/correct_imu_timestamp.cpp`：修正 IMU 时间戳。
    - `tools/edit_binary/edit_timestamp/src/create_iono_parameter.cpp`：生成电离层参数。
    - `tools/edit_binary/edit_timestamp/src/cut_files.cpp`：裁剪数据文件。
    - `tools/edit_binary/edit_timestamp/src/edit_timestamp_utility.cpp`：工具函数实现。
- `tools/edit_binary/generate_replay_tag/`：生成 replay tag 工具。
    - `tools/edit_binary/generate_replay_tag/include/utility.h`：工具函数声明。
    - `tools/edit_binary/generate_replay_tag/option/master_and_client.yaml`：主从模式示例配置。
    - `tools/edit_binary/generate_replay_tag/option/single.yaml`：单机示例配置。
    - `tools/edit_binary/generate_replay_tag/src/generate_tag.cpp`：tag 生成实现。
    - `tools/edit_binary/generate_replay_tag/src/utility.cpp`：工具函数实现。
- `tools/edit_binary/modify_replay_tag/`：修改 replay tag 工具。
    - `tools/edit_binary/modify_replay_tag/src/shift_tag.cpp`：tag 平移/偏移实现。

#### `tools/evaluation/`（评估/对齐/格式转换）

- `tools/evaluation/alignment/`：NMEA/轨迹对齐与外参估计。
    - `tools/evaluation/alignment/include/nmea_formator.h`：NMEA 格式化声明。
    - `tools/evaluation/alignment/src/nmea_align_timestamp.cpp`：时间戳对齐。
    - `tools/evaluation/alignment/src/nmea_estimate_pose_extrinsics.cpp`：外参估计。
    - `tools/evaluation/alignment/src/nmea_formator.cpp`：NMEA 格式化实现。
    - `tools/evaluation/alignment/src/nmea_pose_to_pose.cpp`：pose→pose 变换/导出。
    - `tools/evaluation/alignment/src/nmea_pose_to_position.cpp`：pose→position 导出。
- `tools/evaluation/format_converters/`：多种数据格式互转（源码 + `build/` 产物）。
    - `tools/evaluation/format_converters/CMakeLists.txt`：构建脚本。
    - `tools/evaluation/format_converters/include/geodetic_coordinate.h`：坐标定义。
    - `tools/evaluation/format_converters/include/nmea_formator.h`：NMEA 格式化声明。
    - `tools/evaluation/format_converters/src/*.cpp`：IE/NMEA/TUM/IMR/ImagePack/ImuPack 等互转实现。
    - `tools/evaluation/format_converters/build/*`：该子工具的构建产物与可执行（可删后重建）。
- `tools/evaluation/integrity/`：完整性评估。
    - `tools/evaluation/integrity/evaluate_integrity.py`：评估脚本。
    - `tools/evaluation/integrity/Ground-Truth.tum`：示例真值轨迹。
    - `tools/evaluation/integrity/*.png/*.txt/*.tum`：评估图表与输出。

#### `tools/matlab_plot/`（Matlab 绘图）

- `tools/matlab_plot/geoFunctions/*.m`：地理/坐标相关函数。
- `tools/matlab_plot/plot_ambiguity.m`：模糊度相关绘图。
- `tools/matlab_plot/plot_ie_error.m`：IE 误差绘图。
- `tools/matlab_plot/plot_ionosphere.m`：电离层绘图。
- `tools/matlab_plot/plot_phaserange_residual.m`：载波相位残差绘图。
- `tools/matlab_plot/plot_pseudorange_residual.m`：伪距残差绘图。

#### `tools/ros/`（独立 catkin 工具工作区）

- `tools/ros/src/gici_messages/`：消息包（`CMakeLists.txt`、`package.xml`）。
- `tools/ros/src/gici_tools/`：工具包（`CMakeLists.txt`、`package.xml`）。
- `tools/ros/build/`、`tools/ros/devel/`：catkin 生成物（包含若干 `lib*.so`，可删后重建）。

### 3.7 完整性测试（`integrity_test/`）

- `integrity_test/CMakeLists.txt`：测试工程构建脚本。
- `integrity_test/main.cpp`：测试主入口。
- `integrity_test/include/integrity.h`：测试/算法接口。
- `integrity_test/src/integrity.cpp`：测试/算法实现。
- `integrity_test/test.cpp`：测试驱动/入口之一。
- `integrity_test/test/exclude.cpp`：测试代码（排除/筛选相关）。
- `integrity_test/test/integrity_test.cpp`：完整性测试用例。
- `integrity_test/test/PL_test.cpp`：PL 相关测试。
- `integrity_test/test/pro.cpp`：实验/原型测试。
- `integrity_test/test/sigma_test copy.cpp`：sigma 测试（拷贝版）。
- `integrity_test/test/sigma_test.cpp`：sigma 测试。
- `integrity_test/test/subsets_test.cpp`：子集/组合测试。
- `integrity_test/test/test.cpp`：通用测试。
- `integrity_test/test/*.exe`：已构建的测试可执行（生成物；可删后重编）。

### 3.8 ROS 封装（`ros_wrapper/`）

说明：`ros_wrapper/build/`、`ros_wrapper/devel/` 通常很大且为生成物；核心源代码在 `ros_wrapper/src/gici/`。

- `ros_wrapper/src/gici/CMakeLists.txt`：ROS 包构建脚本。
- `ros_wrapper/src/gici/package.xml`：ROS 包元信息。
- `ros_wrapper/src/gici/msg/GlonassEphemeris.msg`：GLONASS 星历消息定义。
- `ros_wrapper/src/gici/msg/GnssAntennaPosition.msg`：GNSS 天线位置消息定义。
- `ros_wrapper/src/gici/msg/GnssEphemerides.msg`：星历集合消息定义。
- `ros_wrapper/src/gici/msg/GnssEphemeris.msg`：星历消息定义。
- `ros_wrapper/src/gici/msg/GnssIonosphereParameter.msg`：电离层参数消息定义。
- `ros_wrapper/src/gici/msg/GnssObservation.msg`：GNSS 单条观测消息定义。
- `ros_wrapper/src/gici/msg/GnssObservations.msg`：GNSS 观测集合消息定义。
- `ros_wrapper/src/gici/msg/GnssSsrCodeBiases.msg`：SSR 码偏差集合消息定义。
- `ros_wrapper/src/gici/msg/GnssSsrCodeBias.msg`：SSR 码偏差消息定义。
- `ros_wrapper/src/gici/msg/GnssSsrEphemerides.msg`：SSR 星历集合消息定义。
- `ros_wrapper/src/gici/msg/GnssSsrEphemeris.msg`：SSR 星历消息定义。
- `ros_wrapper/src/gici/msg/GnssSsrPhaseBiases.msg`：SSR 相位偏差集合消息定义。
- `ros_wrapper/src/gici/msg/GnssSsrPhaseBias.msg`：SSR 相位偏差消息定义。
- `ros_wrapper/src/gici/option/publish_data_to_ros_topics.yaml`：发布到 ROS topic 的配置。
- `ros_wrapper/src/gici/option/ros_real_time_estimation_*.yaml`：ROS 模式实时估计配置族。
- `ros_wrapper/src/gici/rviz/gici_gic.rviz`：RViz 配置。
- `ros_wrapper/src/gici/src/gici_ros_main.cpp`：ROS 入口。
- `ros_wrapper/src/gici/src/ros_interface/ros_node_handle.cpp`：ROS 侧 NodeHandle。
- `ros_wrapper/src/gici/src/ros_interface/ros_publisher.cpp`：ROS 发布器实现。
- `ros_wrapper/src/gici/src/ros_interface/ros_stream.cpp`：ROS 流/订阅桥接实现。

### 3.9 第三方依赖（`third_party/`）

- `third_party/fast/`：FAST 特征相关库（含 `include/`、`src/`、`lib/`、`test/`、`README.md`）。
- `third_party/rpg_svo/`：RPG SVO 相关库（含 `include/`、`src/`、`lib/`、`test/`）。
- `third_party/rpg_vikit/`：VIKIT 工具库（`vikit_common/`、`vikit_py/`、`vikit_ros/`）。
- `third_party/rtklib/`：RTKLIB（含 `include/`、`src/`、`lib/`）。
- `third_party/segmentation/`：分割模型与 Python 封装（FastSAM/SAM2/YOLO-seg 等权重 + wrapper）。
    - `third_party/segmentation/segmentation_wrapper.py`：分割推理封装。
    - `third_party/segmentation/reexport_onnx.py`：ONNX 导出/重导脚本。
    - `third_party/segmentation/test_wrapper.py`：wrapper 测试脚本。
    - `third_party/segmentation/test_download_pt.py`：权重下载/校验脚本。
    - `third_party/segmentation/*.onnx`、`third_party/segmentation/*.pt`：模型权重/ONNX 文件（体积大，通常不改动）。
    - `third_party/segmentation/images/`、`third_party/segmentation/results/`：示例输入/输出。

### 3.10 实验输出与离线分析（`results/`、`visual_ism/`）

#### `results/`

- `results/compare_results.py`：对比结果脚本。
- `results/extract_log_data.py`：从日志提取数据脚本。
- `results/jacobian_visualize.py`：雅可比可视化脚本。
- `results/sig2_visualization.py`：sig2 可视化脚本。
- `results/sig2intacc_debug.py`、`results/sigma_debug.py`、`results/W_rem_debug.py`：调试脚本。
- `results/visualize_complexity.py`：复杂度可视化脚本。
- `results/weight_matrix_visualize.py`：权重矩阵可视化脚本。
- `results/factor_graph-1212.dot`：图结构导出（Graphviz）。
- `results/*.png`：分析/对比图表（Jacobian/权重矩阵/复杂度/子集等）。
- `results/*.txt`：中间结果/调试输出（subset/sig2 等）。
- `results/*.log.*`：运行日志。
- `results/debug/`：调试目录（可能为空或按实验生成）。
- `results/mask_feat/`：mask+feature 相关逐帧输出（大量 png）。

#### `visual_ism/`

- `visual_ism/check_large_phi_errors.py`：phi 异常检查。
- `visual_ism/compare_qk_results.py`：qk 结果对比。
- `visual_ism/fcdf_overbounding.py`：FCDF overbounding 分析。
- `visual_ism/fit_qi_envelope.py`：QI envelope 拟合。
- `visual_ism/paired_overbounding.py`：成对 overbounding 分析。
- `visual_ism/process_pairwise_sd_errors.py`：pairwise SD 误差处理。
- `visual_ism/qk_fitting_params1e-3.json`：拟合参数。
- `visual_ism/qk_results*.json`：拟合/实验结果（不同阈值/配置）。
- `visual_ism/*.png`：分析图表（pairwise 分布、phi 分析、sigma trend 等）。
