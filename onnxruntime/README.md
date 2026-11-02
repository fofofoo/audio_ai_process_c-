# ONNX Runtime 本地库（x86_64 1.23.2 / Jetson 1.18.0）

本目录托管项目使用的两套 ONNX Runtime，供根目录 `CMakeLists.txt` 按架构自动引用。
两套库的版本、平台和运行期依赖不同，**不要混用头文件或 `.so`**。

| 平台 | 版本 | 目录 | 来源 | 运行时依赖 | 状态 |
|---|---|---|---|---|---|
| x86_64（WSL / PC） | 1.23.2 | `install/onnxruntime-1.23.2/` | 官方预编译 GPU 包 | CUDA 12.x + cuDNN 9.x | 已就绪 |
| aarch64（Jetson Orin NX） | 1.18.0 | `install/onnxruntime-1.18.0/` | 源码编译 | CUDA 11.4 + cuDNN 8.x | 待设备上编译安装 |

目录布局：

```
onnxruntime/
├── README.md
└── install/
    ├── onnxruntime-1.23.2/          # x86_64 官方预编译包（自带 README.md / Privacy.md）
    │   ├── include/
    │   └── lib/
    └── onnxruntime-1.18.0/          # aarch64，Jetson 上编译后安装到这里
        ├── include/
        └── lib/
```

---

## 1. 项目内 CMake 引用

项目根 `CMakeLists.txt` 已按 `CMAKE_SYSTEM_PROCESSOR` 自动选择：

- `aarch64` → `install/onnxruntime-1.18.0`
- `x86_64` → `install/onnxruntime-1.23.2`

推理目标只需调用统一入口：

```cmake
add_executable(your_infer_target.out ...)
ort_link_target(your_infer_target.out)
```

`ort_link_target()` 会完成三件事：

- `target_include_directories` 指向该平台的 `include/`
- `target_link_libraries` 链接 `lib/libonnxruntime.so`
- 设置 `BUILD_RPATH` / `INSTALL_RPATH` 指向 `lib/`，运行时能找到
  `libonnxruntime_providers_*.so`（ORT 运行期用 dlopen 加载）

ORT 未安装时 CMake 只给 WARNING，不中断现有音频链路构建；调用
`ort_link_target()` 的目标才会报错。

---

## 2. x86_64 / WSL：ONNX Runtime 1.23.2

### 2.1 版本与文件

| 项目 | 取值 |
|---|---|
| 版本 | 1.23.2 |
| Git commit | `a83fc4d58cb48eb68890dd689f94f28288cf2278` |
| 架构 | x86_64 |
| 主库 | `lib/libonnxruntime.so.1.23.2` |
| Providers | `libonnxruntime_providers_cuda.so`、`libonnxruntime_providers_shared.so`、`libonnxruntime_providers_tensorrt.so`（可选） |

`install/onnxruntime-1.23.2/README.md` 是官方发布包自带的原始说明，本节记录
项目内的接入方式。

### 2.2 运行时依赖

CUDA Execution Provider 需要：

- **CUDA 12.x**（本机为 `/usr/local/cuda-12.8`）
- **cuDNN 9.x**（与 CUDA 12 匹配）

系统 apt 安装的 cuDNN 8.9.7 不能替代：`libonnxruntime_providers_cuda.so` 的
`NEEDED` 直接是 `libcudnn.so.9`，与 8.x 的 soname 不同。

TensorRT EP 只有在显式启用时才需要 TensorRT 库；只用 CUDA EP 可以忽略
`libonnxruntime_providers_tensorrt.so`。

### 2.3 cuDNN 9 的位置与使用

cuDNN 9 **不安装进系统路径**，避免覆盖系统里的 8.9.7。当前已安装版本
**9.10.2**（`cudnn-linux-x86_64-9.10.2.21_cuda12-archive.tar.xz`，SHA256 已与
NVIDIA 官方清单核对一致），位置：

```
/home/liyun/code/AI/3rd/cudnn-9.10/
├── include/
└── lib/            # libcudnn.so.9、libcudnn_engines_precompiled.so.9 等
```

运行任何使用 CUDA EP 的程序前设置：

```bash
export LD_LIBRARY_PATH=/home/liyun/code/AI/3rd/cudnn-9.10/lib:$LD_LIBRARY_PATH
```

> 不要写进全局 `.bashrc`：会影响所有进程（例如其他 Python 环境可能错误加载
> cuDNN 9）。建议只在 ORT 程序自己的启动脚本里设置。

启动脚本示例（放在项目根，按需改名）：

```bash
#!/bin/bash
# run_infer.sh —— 带 cuDNN 9 环境运行 ORT 推理程序
export LD_LIBRARY_PATH=/home/liyun/code/AI/3rd/cudnn-9.10/lib:$LD_LIBRARY_PATH
exec "$@"
```

```bash
./run_infer.sh ./build/your_infer_target.out --model xxx.onnx
```

**为什么必须用 `LD_LIBRARY_PATH`**：`libonnxruntime_providers_cuda.so` 没有
RUNPATH，只把 cuDNN 9 的 `.so` 放进 ORT 的 `lib/` 目录不够，运行时仍会报
`libcudnn.so.9 => not found`。

### 2.4 验证

```bash
export LD_LIBRARY_PATH=/home/liyun/code/AI/3rd/cudnn-9.10/lib:$LD_LIBRARY_PATH

ldd onnxruntime/install/onnxruntime-1.23.2/lib/libonnxruntime_providers_cuda.so \
    | grep -E "cudnn|cudart|not found"
```

期望：

```
libcudnn.so.9 => /home/liyun/code/AI/3rd/cudnn-9.10/lib/libcudnn.so.9
libcudart.so.12 => /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12
```

不应出现 `not found`。程序内可通过日志确认 `CUDAExecutionProvider` 注册成功；
CUDA EP 创建失败时 ORT 会回退 CPU，推理仍可能成功但速度明显变慢。

### 2.5 常见问题（WSL）

| 现象 | 原因与处理 |
|---|---|
| `libcudnn.so.9: cannot open shared object file` | 没设 `LD_LIBRARY_PATH`，或路径不是 `cudnn-9.10/lib` |
| 设了环境变量仍加载到 `libcudnn.so.8` | `LD_LIBRARY_PATH` 没有排在系统路径前面；用 `ldd` 复核 |
| `libonnxruntime_providers_cuda.so: cannot open` | 主库与 provider 必须在同一 `lib/` 目录；RPATH 未生效时用 `LD_LIBRARY_PATH` 补上 |
| TensorRT provider 报 `libnvinfer.so.*` 找不到 | 没装 TensorRT；不用 TRT EP 时忽略该 provider 文件 |
| CUDA EP 创建失败但程序没崩 | ORT 默认回退 CPU EP；检查 provider 注册与日志 |

---

## 3. Jetson Orin NX：ONNX Runtime 1.18.0

### 3.1 适用平台与选型原因

| 项目 | 取值 |
|---|---|
| 设备 | Jetson Orin NX 16GB |
| 系统 | JetPack 5.1.2（Ubuntu 20.04） |
| 架构 | aarch64，compute capability **sm_87** |
| CUDA | 11.4，`/usr/local/cuda-11.4` |
| cuDNN | 8.6.0（JetPack 自带，`/usr/include` + `/usr/lib/aarch64-linux-gnu`） |
| 编译器 | gcc / g++ 9.4 |
| Python | 3.8 |

选 1.18.0 而不是更新的版本：

- ORT 1.19 起默认分发 CUDA 12 版本，JetPack 5 是 CUDA 11.4，版本对不上；
- 1.18.0 是最后一批官方明确支持 CUDA 11.x 的版本；
- 1.18.1 起 cuDNN 要求跳到 9.x，因此锁定 **1.18.0**（该版本要求 cuDNN 8.9，
  与 JetPack 自带的 8.6 存在小版本差，见 3.6 节）。

> WSL 是 x86_64 机器，无法生成 aarch64 库，也不能用普通交叉编译代替
> （JetPack 5 的 CUDA 11.4 只有 aarch64 原生工具链）。本节所有命令都在
> Jetson 上执行。

### 3.2 前置条件

```bash
# CUDA
ls /usr/local/cuda-11.4/bin/nvcc
nvcc --version          # 应为 release 11.4

# cuDNN
grep CUDNN_MAJOR /usr/include/cudnn_version.h   # 应为 8

# 架构
uname -m                # 应为 aarch64

# 磁盘（源码 + 构建目录 + 依赖约需 25~30 GB）
df -h ~
```

JetPack 5.1.2 自带的 CMake 是 3.16，而 ORT 1.18.0 要求 **CMake ≥ 3.26**，
必须先升级。

内存 16 GB 编译够用，但链接阶段（LTO）峰值较高，建议临时增加 8~16 GB swap：

```bash
swapon --show
# 若不足，临时创建 8 GB（示例）
sudo fallocate -l 8G /swapfile_ort
sudo chmod 600 /swapfile_ort
sudo mkswap /swapfile_ort
sudo swapon /swapfile_ort
```

### 3.3 编译步骤

#### 3.3.1 升级 CMake

```bash
python3 -m pip install --upgrade pip
python3 -m pip install "cmake==3.28.3"
cmake --version    # 应 >= 3.26
```

> PyPI 的 cmake 包提供 aarch64 wheel；若 pip 源不可用，也可从
> https://cmake.org/download/ 下载 `cmake-3.28.3-linux-aarch64.tar.gz` 解压后加入 PATH。

#### 3.3.2 安装 Python 构建依赖

```bash
python3 -m pip install "numpy==1.24.4" onnx protobuf flatbuffers packaging sympy
```

Python 3.8 下 numpy 最高可用 1.24.4，不要装 numpy 2.x。

#### 3.3.3 获取源码

必须用完整 git 仓库（带子模块）。WSL 上那份解压目录缺 `.git` 和
`cmake/external/onnx` 子模块，不能直接拷过去用。

```bash
cd ~
git clone --recursive --depth 1 -b v1.18.0 \
    https://github.com/microsoft/onnxruntime.git onnxruntime-1.18.0
```

国内网络如需代理，在 clone 与后续 `--update` 阶段都要能访问 GitHub
（CMake 会继续下载 onnx 1.16.0、protobuf、eigen、cutlass 等依赖）。

#### 3.3.4 编译

```bash
cd ~/onnxruntime-1.18.0

./build.sh --config Release \
  --build_shared_lib \
  --use_cuda \
  --cuda_home /usr/local/cuda-11.4 \
  --cudnn_home /usr \
  --cuda_version 11.4 \
  --disable_types float8 \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 \
  --parallel 6 \
  --skip_tests \
  --update --build
```

关键参数说明：

| 参数 | 作用 |
|---|---|
| `--build_shared_lib` | 生成 `libonnxruntime.so`（默认只出静态库） |
| `--use_cuda` | 启用 CUDA Execution Provider，同时链接 cuDNN |
| `--cuda_home /usr/local/cuda-11.4` | JetPack 5 的 CUDA 路径 |
| `--cudnn_home /usr` | JetPack 的 cuDNN 头文件在 `/usr/include`、库在 `/usr/lib/aarch64-linux-gnu` |
| `--cuda_version 11.4` | 让构建脚本按 CUDA 11.4 做特性开关检查 |
| `--disable_types float8` | **必须**。CUDA 11.4 < 11.8，没有 FP8 类型；不加会报 `Float 8 types require CUDA>=11.8` |
| `CMAKE_CUDA_ARCHITECTURES=87` | Orin NX 的 compute capability；显式指定后不再走 ORT 默认的多架构列表，编译更快 |
| `--parallel 6` | Orin NX 8 核，留出余量；内存紧张时降到 4 |
| `--skip_tests` | 跳过单元测试，节省 30 分钟以上 |
| `--update --build` | 先配置（下载第三方依赖）再编译 |

> 若设备是 Xavier 系列，把 `87` 改成 `72`；Nano 改成 `53`。

编译约 1.5~3 小时（取决于 Orin NX 散热与 swap 情况）。

### 3.4 安装到本目录

在 Jetson 上、且项目仓库已同步到设备后执行（把 `<项目根>` 换成实际路径，
例如 `/home/nvidia/audio_ai_process`）：

```bash
ORT_SRC="$HOME/onnxruntime-1.18.0"
BUILD="$ORT_SRC/build/Linux/Release"
DEST="<项目根>/onnxruntime/install/onnxruntime-1.18.0"

# 1) 头文件：照官方发布包的平铺布局
mkdir -p "$DEST/include/core" "$DEST/lib"
cp "$ORT_SRC"/include/onnxruntime/core/session/*.h "$DEST/include/"
cp -r "$ORT_SRC"/include/onnxruntime/core/providers "$DEST/include/core/"

# 2) 共享库：-a 保留 .so 符号链接
cp -a "$BUILD"/libonnxruntime.so* "$DEST/lib/"
cp -a "$BUILD"/libonnxruntime_providers_shared.so "$DEST/lib/"
cp -a "$BUILD"/libonnxruntime_providers_cuda.so "$DEST/lib/"
```

安装后的目录结构（CMake 只认这套路径）：

```
install/onnxruntime-1.18.0/
├── include/
│   ├── onnxruntime_c_api.h
│   ├── onnxruntime_cxx_api.h
│   ├── onnxruntime_cxx_inline.h
│   ├── onnxruntime_float16.h
│   ├── onnxruntime_lite_custom_op.h
│   ├── onnxruntime_run_options_config_keys.h
│   ├── onnxruntime_session_options_config_keys.h
│   ├── cpu_provider_factory.h
│   ├── provider_options.h
│   └── core/providers/...
└── lib/
    ├── libonnxruntime.so -> libonnxruntime.so.1.18.0
    ├── libonnxruntime.so.1 -> libonnxruntime.so.1.18.0
    ├── libonnxruntime.so.1.18.0
    ├── libonnxruntime_providers_shared.so
    └── libonnxruntime_providers_cuda.so
```

安装后可删掉源码和构建目录释放空间（约 20 GB），库文件本身约 400~800 MB。

### 3.5 验证

检查动态库依赖：

```bash
DEST="<项目根>/onnxruntime/install/onnxruntime-1.18.0"

ldd "$DEST/lib/libonnxruntime_providers_cuda.so" \
    | grep -E "cudnn|cuda|not found"
```

不应出现 `not found`；应能看到 `libcudnn.so.8`、`libcudart.so.11.0` 等。

用 CUDA EP 跑一个模型：

```bash
ORT_SRC="$HOME/onnxruntime-1.18.0"
DEST="<项目根>/onnxruntime/install/onnxruntime-1.18.0"

export LD_LIBRARY_PATH="$DEST/lib:/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH"
"$ORT_SRC/build/Linux/Release/onnxruntime_perf_test" \
    -e cuda -I -r 3 -m times /path/to/model.onnx
```

能正常输出推理耗时且不报 `no kernel image is available` 即说明 CUDA EP 可用。

### 3.6 常见问题（Jetson）

| 现象 | 原因与处理 |
|---|---|
| `Float 8 types require CUDA>=11.8` | 漏了 `--disable_types float8`；与 `--cuda_version 11.4` 必须同时出现 |
| CMake 报版本过低 | JetPack 自带 3.16，按 3.3.1 升级到 ≥ 3.26 |
| 链接阶段被杀（OOM） | 增大 swap，`--parallel` 降到 4 |
| 编译报找不到 cuDNN 符号 | JetPack 5.1.2 自带 cuDNN 8.6，低于 ORT 1.18.0 要求的 8.9；可到 NVIDIA 开发者网站下载 JetPack 5 对应的 cuDNN 8.9 aarch64 deb 升级 |
| 运行时报 `CUDNN_STATUS_NOT_SUPPORTED` | 同上，优先升级到 cuDNN 8.9；也可临时改用 CPU EP 验证模型本身 |
| `no kernel image is available for execution on the device` | 架构没对上：确认编译时 `CMAKE_CUDA_ARCHITECTURES=87`，且没有链接到 x86 的库 |
| 运行时找不到 `libonnxruntime_providers_cuda.so` | 该文件与 `libonnxruntime.so` 必须在同一目录（或 LD_LIBRARY_PATH 中），ORT 运行时用 dlopen 加载 |

---

## 4. 两平台差异速查

| 项目 | x86_64（WSL） | aarch64（Jetson Orin NX） |
|---|---|---|
| 版本 | 1.23.2 | 1.18.0 |
| 来源 | 官方预编译 | 源码编译 |
| CUDA | 12.8 | 11.4（JetPack 5.1.2） |
| cuDNN | 9.10.2（`/home/liyun/code/AI/3rd/cudnn-9.10`） | 8.x（JetPack 自带） |
| 计算能力 | 按预编译包（含 sm_86/89/90 等） | sm_87 |
| 运行期额外环境 | `LD_LIBRARY_PATH` 指向 cuDNN 9 | 通常无需额外设置 |
| 头文件版本 | 1.23.2 | 1.18.0 |

两套头文件版本不同，但这一段 C/C++ API 向后兼容；不要跨平台混用头文件或 `.so`。

---

## 5. 版本记录

| 日期 | 说明 |
|---|---|
| 2026-11-02 | 创建；合并原 `install/onnxruntime-1.18.0/README.md` 与 `USAGE.md`，统一放在本文件 |
| 2026-11-02 | 目标设备确认 Jetson Orin NX 16GB（sm_87）；WSL 侧 cuDNN 9.10.2 安装完成并校验 |
