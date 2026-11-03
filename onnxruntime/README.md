# ONNX Runtime 本地库（x86_64 1.23.2 / Jetson 1.18.0）

本目录托管项目使用的两套 ONNX Runtime，供根目录 `CMakeLists.txt` 按架构自动引用。
两套库的版本、平台和运行期依赖不同，**不要混用头文件或 `.so`**。

| 平台 | 版本 | 目录 | 来源 | 运行时依赖 | 状态 |
|---|---|---|---|---|---|
| x86_64（WSL / PC） | 1.23.2 | `install/onnxruntime-1.23.2/` | 官方预编译 GPU 包 | CUDA 12.x + cuDNN 9.x | 已就绪 |
| aarch64（Jetson Orin NX） | 1.18.0 | `install/onnxruntime-1.18.0/` | 源码编译 | CUDA 11.4 + cuDNN 8.x | 已在 Jetson 上编译通过（2026-11-03） |

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
| Python | ≥ 3.8（系统自带 3.8 即可；推荐独立 conda 环境，见 3.3.1） |

选 1.18.0 而不是更新的版本：

- ORT 1.19 起默认分发 CUDA 12 版本，JetPack 5 是 CUDA 11.4，版本对不上；
- 1.18.0 是最后一批官方明确支持 CUDA 11.x 的版本；
- 1.18.1 起 cuDNN 要求跳到 9.x，因此锁定 **1.18.0**（该版本要求 cuDNN 8.9，
  与 JetPack 自带的 8.6 存在小版本差，见 3.6 节）。

**当前选 1.18.0 的困难（本次实测）**：官方 CI/打包镜像的最低 CUDA 是 11.8，而 JetPack 5
只有 11.4，nvcc 前端更旧，编译时需要手工处理四处源码级问题——3.3.6 的三处
`missing return statement`（`common.cuh`、`max_pool_with_index.cu`、CUTLASS MoE）和
3.3.7 的 `resize_antialias_impl.cu` cicc 错误。四处都不是代码缺陷，改动量很小
（两处各补一行 return、一处换成两个重载、一处替换文件），但必须知道位置，否则
会在 53%（`common.cuh`）、58%（`resize_antialias_impl.cu`）、62%（`max_pool_with_index.cu`）、
75%（CUTLASS MoE）附近反复中断。

**换来的优势**：1.18.0 与 JetPack 5 的 CUDA 11.4 / cuDNN 8.x 同代，Jetson 上运行时
不需要像 x86 侧那样额外挂 cuDNN 9；算子覆盖比 1.17.0 全（含 CUDA 版 Resize
antialias）；C/C++ API 与 x86 侧 1.23.2 一致，项目里同一套调用代码不用按平台分叉。
本次编译完成后已在真机验证：`onnxruntime_test_all --gtest_filter='*Cuda*:*CUDA*'`
跑的 29 个用例全部通过，说明 CUDA EP 在 Jetson 上能正常执行，不只是编得过。

**备选：1.17.0（源码层面零补丁）**

先看两个版本的 CUDA 口径对照（直接查源码，不是凭印象）：

| 版本 | CMake 硬性下限检查 | 官方 CI / 打包镜像 | 对本机 CUDA 11.4 的含义 |
|---|---|---|---|
| 1.18.0 | **有**：`CMAKE_CUDA_COMPILER_VERSION < 11.4` 直接 `FATAL_ERROR`（`cmake/CMakeLists.txt`） | CUDA **11.8.0** + cuDNN 8（`tools/ci_build/github/linux/docker/Dockerfile.manylinux2_28_cuda`）；`dockerfiles/Dockerfile.cuda` 另用 12.1.1 | 11.4 是能过配置的最低值，但低于官方口径，源码层问题要自己补 |
| 1.17.0 | **无**：只有 `< 11.6` 时关掉 flash attention（`STATUS`，不报错） | 同样是 CUDA **11.8.0** + cuDNN 8；TensorRT 支线另有 11.6 镜像 | 没有硬性拦截，11.4 仍属低于官方口径，但不会在 CMake 阶段被拒 |

结论：**两个版本的官方口径都是 CUDA 11.8**，1.18.0 额外用 `FATAL_ERROR` 把 11.4 定为
硬下限；编译前先查这两处（CMake 的版本检查 + `Dockerfile.*` 的 CI 镜像），比出问题再
翻日志快得多。`--disable_types float8` 的 11.8 判定两版也一样（不传 `--cuda_version`
就不触发）。

1.18.0 在 CUDA 11.4 下需要 3.3.6 的三处 `missing return statement` 修改和 3.3.7 的
一个替换文件。如果换机重装想省掉这些，可以改用 **1.17.0**：逐文件对照 1.17.0 源码后
确认，这四处补丁对应的代码在 1.17.0 中都不存在或写法不同。

| 1.18.0 需要处理的位置 | 1.17.0 的对应实现 | 是否需要补丁 |
|---|---|---|
| `cu_inc/common.cuh` 的 `_IsInf`（`IsInfTyped` 模板） | 没有 `IsInfTyped`，该段是 1.18 新增 | 否 |
| `nn/max_pool_with_index.cu` 的布局分派 lambda | 只有单模板 `MaxPoolWithIndexKernel`，无 `if constexpr` | 否 |
| CUTLASS `moe/cutlass_extensions/.../moe_cutlass_kernel.h` 的 `CreateMMA` lambda | 是 `moe/ft_moe/moe_cutlass_kernel.h`，第 388 行直接 `Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);` | 否 |
| `tensor/resize_antialias_impl.cu` 的 cicc LLVM IR 错误 | 没有 CUDA 版 antialias 实现，该文件不存在（只有 CPU 的 `upsample_antialias.h`） | 否 |

代价与限制：

- **没有 CUDA 版 Resize antialias**。模型里 `Resize` 带 `antialias=1` 时该节点会回退
  CPU，结果正确但会切分 CUDA 图、多一次拷贝；当前 ASR 三模型不涉及。
- **省不掉 gcc-10**：1.17.0 的 `cmake/CMakeLists.txt` 有与 1.18.0 完全相同的
  `check_cxx_compiler_flag(-march=armv8.2-a+bf16)` + `FATAL_ERROR` 检查，CMake ≥ 3.26、
  `--skip_submodule_sync`、`onnxruntime_BUILD_UNIT_TESTS=OFF` 也都一样。
- 依赖清单不同，`fetch_ort_deps.py` 要按 1.17.0 自己的 `cmake/deps.txt` 重新预下载。
- **未在真机实编验证**：以上是源码对照结论。1.17.0 的 CUDA 目录里 `if constexpr`
  仅 10 处、其中唯一的 `.cu` 是 `softmax_impl.cu`（写法与 1.18.0 相同，本次没报错），
  同类型的误报概率明显更低；但 CUTLASS 模板在 CUDA 11.4 下仍可能有未知风险。

因此主线仍锁 1.18.0（补丁已实测跑通），1.17.0 只作为换机重装或 1.18.0 再遇到新
nvcc 问题时的退路；切换时要同步改根 `CMakeLists.txt` 的 ORT 路径与本 README。

> WSL 是 x86_64 机器，无法生成 aarch64 库，也不能用普通交叉编译代替
> （JetPack 5 的 CUDA 11.4 只有 aarch64 原生工具链）。本节所有命令都在
> Jetson 上执行。

### 3.2 前置条件（设备实测）

本节记录 Jetson Orin NX 16GB + JetPack 5.1.2 上的实测环境，并把「硬性要求」与
「本机取值」分开写。判断依据来自 ORT 1.18.0 源码（`tools/ci_build/build.py`、
`cmake/CMakeLists.txt`），不是经验估计。

```bash
# 以下命令在未激活 conda 的系统 shell 里执行

# 架构（应为 aarch64）
uname -m

# CUDA（应为 11.4）
ls /usr/local/cuda-11.4/bin/nvcc
nvcc --version

# nvcc 允许的最高 GCC 主版本（JetPack 5.1.2 实测为 11，决定能配哪个 GCC）
grep -n "__GNUC__ >" /usr/local/cuda-11.4/include/crt/host_config.h

# cuDNN（major 应为 8；minor 记录实测值）
grep CUDNN_MAJOR /usr/include/cudnn_version.h
grep CUDNN_MINOR /usr/include/cudnn_version.h

# GCC / G++：aarch64 必须支持 -march=armv8.2-a+bf16（GCC ≥ 10），见 3.2.3
gcc --version | head -1
g++ --version | head -1
echo 'int main(){}' > /tmp/t.cpp
g++ -march=armv8.2-a+bf16 -c /tmp/t.cpp -o /tmp/t.o && echo "bf16 OK"

# 系统 GCC 不满足时，替代编译器是否能装（本机实测 10.5.0）
apt-cache policy gcc-10 g++-10

# 系统 libstdc++（gcc-10 编出来的库要在它上面跑，需 ≥ 10）
dpkg -l libstdc++6 | tail -1

# CMake（ORT 1.18.0 要求 >= 3.26；JetPack 5.1.2 自带 3.16.3，不够）
cmake --version

# Python（只要求解释器 >= 3.8）
python3 -V

# 内存与 swap（记录实测值；先不预置 swap）
free -h
swapon --show

# 磁盘（源码 + 构建约 25~30 GB；本方案放 /mnt/data）
df -h ~
df -h /mnt/data
```

源码完整性（`VERSION_NUMBER`、`cmake/external/onnx` 目录存在）无法在获取源码前
检查，统一放到 3.3.3 解压之后验证。

实测结果：

| 项目 | 要求 | 本机实测 | 结论 |
|---|---|---|---|
| 架构 | aarch64 | aarch64 | ✅ |
| CUDA | 11.x（本方案锁 11.4） | 11.4.315，`/usr/local/cuda-11.4` | ✅ |
| nvcc 宿主编译器 | 不低于所选 GCC 主版本 | `host_config.h` 上限 `__GNUC__ > 11` | ✅ gcc-10 可用 |
| cuDNN | 官方口径 8.9 | 8.6.0（major 8 / minor 6） | ⚠️ 可先按 8.6 编，见 3.6 |
| 编译器 | aarch64 必须支持 `-march=armv8.2-a+bf16`（GCC ≥ 10） | gcc / g++ 9.4.0 不支持 bf16；gcc-10 10.5.0 可装 | ⚠️ 装 gcc-10 编，见 3.2.3 |
| libstdc++6 | ≥ 10（配 gcc-10 产物） | 10.5.0-1ubuntu1~20.04 | ✅ |
| CMake | **≥ 3.26** | 系统 3.16.3 | ❌ 必须先另装 3.28.3 |
| Python | **≥ 3.8，且只要解释器** | 系统自带 3.8 | ✅ |
| 内存 | 无硬性下限，先按实测环境直接编 | 15 GiB（可用约 12 GiB） | 先试，OOM 再处理 |
| swap | 暂不预置 | 仅 7.6 GB zram | 先不动，见 3.2.5 |
| 磁盘 | 源码 + 构建约 25~30 GB | `/mnt/data` 可用 389 GB | ✅ 构建放这里 |
| 源码 | 完整发布包（tarball 或 git clone） | 官方 v1.18.0 tarball | ✅ 需加 `--skip_submodule_sync` |

#### 3.2.1 四条硬门槛（不满足会直接失败）

1. **CMake ≥ 3.26。** `cmake/CMakeLists.txt` 第一行是
   `cmake_minimum_required(VERSION 3.26)`；3.16.3 在配置第一步就报错退出。
   不能改源码降版本——构建脚本用到了 3.26 才引入的 CMake 语法，改版本号后
   会在后续配置步骤以其它形式失败。
2. **Python ≥ 3.8。** `tools/ci_build/build.py` 在模块加载时执行
   `_check_python_version()`，低于 3.8 直接抛 `UsageError`；同时
   `cmake/CMakeLists.txt` 在 `--build_shared_lib` 下执行
   `find_package(Python 3.8 COMPONENTS Interpreter)`。JetPack 系统 Python
   （Ubuntu 20.04 自带 3.8）版本上刚好达标；为了不改基础环境，编译推荐在
   独立 conda 环境里进行，见 3.3.1。
3. **源码必须是完整发布包（tarball 或 git clone）。** `--update` 默认会执行
   `git submodule sync/update --recursive`；使用官方 tarball 时没有 `.git`，
   必须加 `--skip_submodule_sync`。onnx 1.16.0、flatbuffers 23.5.26、protobuf、
   eigen、cutlass 等仍由 `cmake/deps.txt` 下载，不依赖子模块目录。零散拷贝的
   残缺目录（缺源文件）仍然不能用。
4. **aarch64 编译器必须支持 `-march=armv8.2-a+bf16`（GCC ≥ 10）。**
   `cmake/CMakeLists.txt` 会对 aarch64 执行
   `check_cxx_compiler_flag(-march=armv8.2-a+bf16 HAS_ARM64_BFLOAT16)`，
   失败即 `FATAL_ERROR`；JetPack 5.1.2 自带的 GCC 9.4 不满足。而且
   `cmake/onnxruntime_mlas.cmake` 会用同一 flag 无条件编译 Sbgemm 内核，
   跳过检查也会在编译期失败。装 gcc-10 的办法见 3.2.3。

#### 3.2.2 Python 库：本构建不要求任何版本

这是最容易误解的一点，结论如下：

- `--build_shared_lib --skip_tests` **不需要** `onnx`、`numpy`、`flatbuffers`、
  `protobuf`、`sympy`、`packaging` 等任何 pip 包。C++ 侧依赖全部来自
  `cmake/deps.txt` 下载的源码包（onnx、protobuf、flatbuffers 等）。
- PyPI 的 `onnx`（Python 模型格式库）只在 ORT 的 Python 测试里被
  `import onnx`（`build.py` 约 2105 行）；导入失败只打印
  「ONNX tests will be skipped」，`--skip_tests` 下根本不会执行。
- `install_python_deps()`（numpy / sympy / packaging / cerberus / pytest 等）
  只在 **Windows + `--enable_pybind`** 分支调用，Linux 构建不会触发。
- `numpy` 只在开启 `--enable_pybind` 时由
  `find_package(Python 3.8 COMPONENTS ... NumPy)` 检查；本方案不开 pybind。
- 因此构建过程不检查这些 pip 包的版本，环境里装没装、版本新旧都无关紧要，
  不需要为了编译去安装或降级。

> 只有以后要跑 ORT 的 Python 测试或构建 Python wheel 时，才需要
> `pip install onnx numpy ...`，并按 ORT 的 Python 包要求选版本。

#### 3.2.3 编译工具：GCC ≥ 10（aarch64 硬门槛）

本机第一次编译在配置阶段直接失败：

```text
CMake Error at CMakeLists.txt:659 (message):
  The compiler doesn't support BFLOAT16!!!
```

原因不在 CMake、不在依赖包，在 GCC 版本：

- ORT **1.17 起**在 `cmake/CMakeLists.txt` 给 aarch64 加了检查：
  `check_cxx_compiler_flag(-march=armv8.2-a+bf16 HAS_ARM64_BFLOAT16)`，
  失败即 `FATAL_ERROR`。1.17 / 1.18 / 1.19 / 1.20 都这么写，没有开关能关掉。
- `+bf16` 是 GCC 10 才加进 AArch64 `-march` 的扩展。JetPack 5.1.2 自带
  GCC 9.4 不认识，实测：

  ```text
  $ g++ -march=armv8.2-a+bf16 -c /tmp/t.cpp
  cc1plus: error: invalid feature modifier ‘bf16’ in ‘-march=armv8.2-a+bf16’
  cc1plus: note: valid arguments are: fp simd crypto crc lse fp16 rcpc rdma dotprod ...
  ```

- 这不只是"检查"：`cmake/onnxruntime_mlas.cmake` 会用同一个 flag 无条件编译
  `SbgemmKernelNeon.S` 与 `sbgemm_kernel_neon.cpp`，所以跳过检查也会在编译期
  以同样原因失败，打补丁没有意义。
- 官方 BUILD 文档没写 aarch64 / Jetson 的 GCC 版本要求，这一条只能从源码检查
  里读出来；本机已实测确认。

解决：装 gcc-10 与 9.4 并存，不改系统默认编译器：

```bash
sudo apt update
sudo apt install -y gcc-10 g++-10
g++-10 -march=armv8.2-a+bf16 -c /tmp/t.cpp -o /tmp/t.o && echo "bf16 OK"
```

兼容性已逐项核对：

| 项目 | 实测 | 结论 |
|---|---|---|
| gcc-10 / g++-10 | focal-updates/universe，10.5.0 | ✅ 可装 |
| CUDA 11.4 nvcc | `host_config.h` 只在 `__GNUC__ > 11` 时报错 | ✅ 10.5 可用 |
| 系统 libstdc++6 | 10.5.0-1ubuntu1~20.04 | ✅ gcc-10 产物可直接运行 |

编译时用三个定义把工具链显式指过去（3.3.5 的命令里同样要加）：

```bash
--cmake_extra_defines CMAKE_C_COMPILER=/usr/bin/gcc-10 \
                      CMAKE_CXX_COMPILER=/usr/bin/g++-10 \
                      CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10
```

换编译器后必须清掉旧构建目录，否则 CMake 缓存里的 `/usr/bin/c++` 与新编译器
冲突，配置直接报 "compiler changed"：

```bash
rm -rf build/Linux/Release
```

备选：不想动工具链可退到 **ORT 1.16.3**——它的 `cmake/CMakeLists.txt` 与
`cmake/onnxruntime_mlas.cmake` 里都没有 bf16 检查与 bf16 编译标志，GCC 9.4
可直接编；代价是要重新准备 1.16.3 那套依赖包，见 3.3.4。

#### 3.2.4 cuDNN 8.6 与 8.9 的取舍

JetPack 5.1.2 自带 cuDNN 8.6.0，ORT 1.18.0 官方口径是 8.9。本方案选择
**先用 8.6 直接编译**：编译期一般不卡版本；运行期若个别算子报
`CUDNN_STATUS_NOT_SUPPORTED`，再按 3.6 升级到 8.9 复测，不必提前替换系统库。

#### 3.2.5 内存与 swap（先不加，OOM 再补）

Orin NX 16GB 的 15 GiB 物理内存先直接编译，不预置磁盘 swap。CUDA kernel 的
LTO 链接阶段峰值较高，若链接阶段被内核杀掉（日志出现 `Killed` 或
`c++: fatal error: Killed signal terminated program`），再按下面步骤临时加
16 GB 磁盘 swap，并可在 3.3.5 把 `--parallel 6` 降到 4 后重跑：

```bash
free -h
swapon --show

sudo fallocate -l 16G /mnt/data/swapfile_ort
sudo chmod 600 /mnt/data/swapfile_ort
sudo mkswap /mnt/data/swapfile_ort
sudo swapon -p 10 /mnt/data/swapfile_ort    # 优先级高于 zram 的 5

# 验证完成后回收
# sudo swapoff /mnt/data/swapfile_ort && sudo rm /mnt/data/swapfile_ort
```

### 3.3 编译步骤

> **后续 3.3.1 ~ 3.3.5 的所有命令都必须在 3.3.1 准备的虚拟 Python 环境里执行**
> （复用已有环境如 `conda activate py310`，新建环境为 `conda activate ort-build`；
> 用 venv 时对应 `source .../bin/activate`）。系统 shell 里的 `cmake` 仍是 3.16.3，
> 未做任何修改；忘记激活会直接卡在 CMake 版本检查，或在错误的环境里安装依赖。

#### 3.3.1 准备 Python ≥ 3.8 环境与 CMake ≥ 3.26

**首选使用虚拟环境，不动 JetPack 基础环境的 CMake。** 已经有 conda 环境时
建议直接复用，不要新建；Python 只要求 ≥ 3.8 且只要解释器，构建不依赖任何
Python 库，所以只需往环境里加一个 CMake，不会升级或降级已有包。先关闭 base 自动激活：

```bash
conda config --set auto_activate_base false   # 必须关闭，否则新 shell 自动激活 base，影响 ROS 等系统包
```

**分支 A：复用已有虚拟环境（推荐）**——把 `py310` 换成实际环境名：

```bash
conda activate py310              # 已有环境，示例名
which python3                     # 应为 <miniconda 安装目录>/envs/py310/bin/python3
conda list cmake                  # 先查 conda 版 cmake；有则 conda remove -n py310 cmake -y
pip install "cmake==3.28.3"       # 只新增 cmake 入口，不动其它包
hash -r                           # 清 bash 缓存，避免仍执行旧 cmake
which cmake && cmake --version    # 应显示 3.28.3

# 失败（仍显示 3.16.3 或 Could not find CMAKE_ROOT）时：
#   conda remove -n py310 cmake -y
#   pip install --force-reinstall "cmake==3.28.3"
#   hash -r
```

**分支 B：没有现成环境时新建**：

```bash
conda create -n ort-build python=3.10 -y
conda activate ort-build
which python3
pip install "cmake==3.28.3"
hash -r
which cmake && cmake --version    # 应显示 3.28.3
```

PyPI 的 `cmake` 包是 CMake 官方 aarch64 二进制的分发形式，安装后只落在
conda 环境的 `bin/` 下，激活该环境时生效；退出环境后系统仍是 3.16.3，
不影响其它项目。若 pip 源不可用，也可从 https://cmake.org/download/ 下载
`cmake-3.28.3-linux-aarch64.tar.gz`，解压后只在编译 shell 里
`export PATH=<解压目录>/bin:$PATH`。设备上没有 conda 时，也可用系统 Python 3.8
建 `venv`，或直接用上面这种 tarball 方式；两者同样不修改系统环境。

#### 3.3.2 Python 库（本构建不需要）

按 3.2.2 的源码结论，`--build_shared_lib --skip_tests` 不要求任何 Python 库，
**不要**执行 `pip install numpy onnx protobuf flatbuffers packaging sympy`。
环境里已有什么包保持原样即可，不需要为编译额外安装或升级。

#### 3.3.3 获取源码

用官方 v1.18.0 发布包（tarball）。归档包不带 `.git`，`cmake/external/onnx` 子模块
为空，但 ORT 构建所需的 onnx 由 `cmake/deps.txt` 通过 FetchContent 下载，不依赖该子模块
（子模块只用于 fuzzing 和 WASM）。所以 tarball 能正常编译，
**但 3.3.5 必须加 `--skip_submodule_sync`**。

```bash
cd /mnt/data/code/3rd
wget https://github.com/microsoft/onnxruntime/archive/refs/tags/v1.18.0.tar.gz
tar -xzf v1.18.0.tar.gz                # 解压出 onnxruntime-1.18.0/
cd onnxruntime-1.18.0

cat VERSION_NUMBER                     # 应为 1.18.0
ls cmake/external/onnx                 # 空目录属于正常
```

wget 被墙挡住时，在 URL 前加镜像前缀即可（如
`https://ghfast.top/https://github.com/...`），或在能下载的机器上下好再拷进来。
第三方依赖不在这里处理，统一走 3.3.4。

#### 3.3.4 预下载依赖

Jetson 现场直连 GitHub 时，CMake 的 FetchContent 下载阶段会被中途掐断
（实测 abseil 成功、date v3.0.1 报 curl `status_code: 56` /
`Failure when receiving data from the peer`）。所以固定成两步：**先在能下载的机器上
把依赖全部下好并校验，随项目拷到 Jetson，再改写 `cmake/deps.txt` 为本地路径**。
这样 3.3.5 的配置阶段完全不需要网络。

镜像用 `https://ghfast.top/`（备选 `https://gh-proxy.com/`、`https://ghproxy.net/`），
下发的是原始文件，SHA1 与 deps.txt 一致；
下载脚本 fetch_ort_deps.py 随仓库提供，onnxruntime源码路径写死在文件顶部，按需修改为本地路径：[fetch_ort_deps.py](fetch_ort_deps.py)。
 fetch_ort_deps.py 脚本具备1.下载cmake依赖、2.修改本地cmake/deps.txt 两个功能，所以本地下载完成后，还需要将下好依赖拷贝过去，再把主控onnxruntime路径中的cmake/deps.txt也改成下好的依赖库路径。


**为什么必须改 `cmake/deps.txt`**：CMake 配置阶段只读源码目录里的这一份
（Jetson 上是 `/mnt/data/code/3rd/onnxruntime-1.18.0/cmake/deps.txt`），
包堆在 `ort_deps/` 里它不会自己去找。deps.txt 的 URL 列本身支持本地路径，
脚本就是把 URL 列改成指向本地文件：

```
date;https://github.com/HowardHinnant/date/archive/refs/tags/v3.0.1.zip;2dac...
  ↓ 改写后（路径按 fetch_ort_deps.py 所在目录生成）
date;<fetch_ort_deps.py 所在目录>/install/onnxruntime-1.18.0/ort_deps/date-v3.0.1.zip;2dac...
```

**要改的是 Jetson 上那份，不是开发机这份**。开发机改写后是 `/home/liyun/...`，
Jetson 上不存在；两边各自跑一遍脚本，各自生成自己的路径。

**第一步：在有VPN的主机上预下载ONNX的CMake依赖**：

```bash
cd <项目根>
python3 -I onnxruntime/fetch_ort_deps.py
```

包落在 `fetch_ort_deps.py` 所在目录的 `install/onnxruntime-1.18.0/ort_deps/`，文件名是
`<依赖名>-<原文件名>`，约 40 个、几百 MB。注意这个目录在 `.gitignore` 里，
**git 不会把包带到 Jetson**。

**第二步：把 `ort_deps/` 物理拷到 Jetson 项目的同名位置，再在 Jetson 上重跑脚本**。
用平时同步项目的方式（rsync / scp / U 盘均可），且要包含被 ignore 的 `install/`：

```
开发机 <fetch_ort_deps.py 所在目录>/install/onnxruntime-1.18.0/ort_deps/
   ↓ 拷贝
Jetson <fetch_ort_deps.py 所在目录>/install/onnxruntime-1.18.0/ort_deps/
```

然后在 Jetson 上执行：

```bash
conda activate py310
python3 -I <项目根>/onnxruntime/fetch_ort_deps.py
```

这一步不下载，只做两件事：逐个校验 SHA1（全部命中就打印 `[已存在]`），
然后把 `/mnt/data/code/3rd/onnxruntime-1.18.0/cmake/deps.txt` 改写成 Jetson 的绝对路径。
**不能直接把开发机改写好的 deps.txt 拷过去**——里面的绝对路径是开发机的，拷过去全部失效。

要点：

- 脚本按存在性自动选源码目录：开发机 `/home/liyun/code/AI/3rd/onnxruntime-1.18.0`，
  Jetson `/mnt/data/code/3rd/onnxruntime-1.18.0`。
- **`ort_deps/` 路径固定为 `<fetch_ort_deps.py 所在目录>/install/onnxruntime-1.18.0/ort_deps/`**，
  与项目根无关。把脚本拷到别处执行（例如直接放进 ORT 源码目录），路径会跟着变成
  `<源码目录>/install/onnxruntime-1.18.0/ort_deps/`，此时把包放在仓库 `onnxruntime/` 下就不会被识别。
  要固定位置就改脚本顶部的 `DEPS_DIR`。只要 `deps.txt` 改写后的路径与实际文件一致，构建都能用。
- 任何一项失败时**不改写** `deps.txt`，换 `MIRROR` 后重跑，已完成的会跳过。
- `eigen` 在 gitlab.com，不走镜像；连不上时单独手动下载放进 `ort_deps/` 再重跑。
- **`eigen` 的 SHA1 已被脚本覆盖**：GitLab 重新生成过归档，同一 commit
  （`e7248b26...`）的 zip 字节变了，ORT 1.18.0 记录的 `be8be39f...` 对不上；
  实测 GitLab 现版 `32b145f5...` 与 GitHub 镜像同 commit 归档解压后文件树零差异，
  只是打包元数据不同。脚本里 `SHA1_OVERRIDES` 已写入实测值并在改写 deps.txt 时带上，
  否则手动下载和 CMake 自下都会校验失败。上游 v1.19/v1.20 仍是旧 hash，
  v1.22 起才换 commit。
- 不想跑第二次时，也可以在 Jetson 上手工把 deps.txt 的 URL 换成
  `ort_deps/<依赖名>-<原文件名>`，效果相同；文件名对不上会漏。
- 首次运行会把原文件备份成 `cmake/deps.txt.orig`；想恢复联网构建用
  `cp cmake/deps.txt.orig cmake/deps.txt`。日常不用改回去——有预下载的本地路径，
  配置阶段就不会再碰网络。

#### 3.3.5 编译

编译约 1.5~3 小时，**命令要写日志**：日志是失败后唯一能回看的东西。
用 NoMachine / 本地桌面时直接在当前终端跑即可（进程在机器上，断开远程桌面不会杀它，
只要不关终端、不注销、不重启）；从 SSH 登录时才需要套一层 `tmux`
（终端复用器，断线后 `tmux attach -t ort-build` 可回到原来的会话）。

**换 gcc 后必须从空的构建目录开始**：之前用 9.4 配置失败时已经生成过
`build/Linux/Release/CMakeCache.txt`，CMake 记住了 `/usr/bin/c++`，再用新编译器
会直接报 "compiler changed"。先删掉（只含失败的配置产物，依赖会从本地
`ort_deps/` 重新解压，不联网），再跑下面的命令。gcc-10 的安装见 3.2.3。

```bash
rm -rf /mnt/data/code/3rd/onnxruntime-1.18.0/build/Linux/Release

conda activate py310              # 换成实际使用的编译环境
cd /mnt/data/code/3rd/onnxruntime-1.18.0

# 仅 SSH 登录时取消注释；NoMachine / 本地桌面直接跳过
# tmux new -s ort-build           # 断线后 tmux attach -t ort-build 回到现场

./build.sh --config Release \
  --build_shared_lib \
  --use_cuda \
  --cuda_home /usr/local/cuda-11.4 \
  --cudnn_home /usr \
  --cuda_version 11.4 \
  --disable_types float8 \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 \
  --cmake_extra_defines CMAKE_C_COMPILER=/usr/bin/gcc-10 \
                        CMAKE_CXX_COMPILER=/usr/bin/g++-10 \
                        CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 \
  --parallel 6 \
  --skip_tests \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
  --skip_submodule_sync \
  --update --build 2>&1 | tee /mnt/data/ort_build_$(date +%m%d_%H%M).log
```

跑过 3.3.4 后，`--update` 的配置阶段读的是本地包，全程不需要网络；如果恢复成
`deps.txt.orig`，则会在这一步联网下载依赖（几百 MB）。日志出现
`-- Configuring done` / `-- Build files have been written to:` 后即进入编译。

配置阶段先确认两行再走开：`HAS_ARM64_BFLOAT16 - Success` 和
`C++ compiler version: 10.5.0`。如果仍显示 9.4.0、或继续报
`The compiler doesn't support BFLOAT16`，说明三个 `-D` 没生效——先查有没有
删掉旧 `build/Linux/Release`，再核对路径 `/usr/bin/gcc-10` 是否存在。

`build.sh` **只编译不安装**：v1.18.0 的 `build.py` 没有 `--install` /
`--install_prefix` 参数（argparse 严格解析，传了直接 `unrecognized arguments` 退出），
产物留在 `build/Linux/Release/`，安装路径在 3.4 再决定。日志里的
`Failed to import psutil` 只是让 build.py 退回 `NVCC_THREADS=1`，不影响结果。

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
| `CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` / `CMAKE_CUDA_HOST_COMPILER` | 把整条工具链指向 `/usr/bin/gcc-10`、`/usr/bin/g++-10`。**必须三个一起给**：前两个决定 C/C++，第三个决定 nvcc 的宿主编译器（不指定会按 PATH 退回 gcc 9）。原因见 3.2.3 |
| `--parallel 6` | Orin NX 8 核，留出余量；内存紧张时降到 4 |
| `--skip_tests` | 只是**不运行**测试，测试目标照样会编译。要和下一行配合才是「既不编也不跑」 |
| `onnxruntime_BUILD_UNIT_TESTS=OFF` | **不编译**测试目标（`onnxruntime_test_all`、`onnxruntime_shared_lib_test`、`onnx_test_runner`、`onnxruntime_perf_test`），省约 10%~15% 的编译时间与磁盘；对最终 `libonnxruntime.so` 的功能没有影响。改这个 CMake 开关会触发一次重新配置，已编好的库目标不会重编 |
| `--skip_submodule_sync` | tarball 没有 `.git`，跳过 `git submodule update`；onnx 仍由 `deps.txt` 下载 |
| `--update --build` | 先配置（下载第三方依赖）再编译 |

> 若设备是 Xavier 系列，把 `87` 改成 `72`；Nano 改成 `53`。

编译时长取决于 Orin NX 散热与是否触发 swap。

> 如果不关单元测试目标，编译到 84% 左右会满屏这样的提示：
> `note: parameter passing for argument of type 'std::pair<float, float>' when C++17 is enabled changed to match C++14 in GCC 10.1`。
> 这是 GCC 10.1 修正 C++17 下 `std::pair` 传参 ABI 的 note（不是 error / warning）。
> 整个 ORT 用同一个 gcc-10、同一标准编译，ABI 一致，可以直接忽略；加上
> `onnxruntime_BUILD_UNIT_TESTS=OFF` 后不再编译测试代码，这类提示也就没了。

#### 3.3.6 CUDA 11.4 的 nvcc 误报 missing return statement（需手工补一行）

**现象**：配置阶段全部通过后，编译到 53% 左右在
`onnxruntime/core/providers/cuda/math/unary_elementwise_ops_impl.cu` 报 12 个错误：

```
error: missing return statement at end of non-void function
"onnxruntime::cuda::_IsInf<T, detect_positive, detect_negative>::operator()
 [with T=float, detect_positive=true, detect_negative=false]"
...
make[2]: *** [.../unary_elementwise_ops_impl.cu.o] Error 1
make[1]: *** [.../onnxruntime_providers_cuda.dir/all] Error 2
```

这不是警告，CUDA provider 直接编不过。日志里后面还在编 CPU 的 `.cc` 是
`--parallel` 的其他并行任务在收尾，最终 `build.py` 仍以失败退出。

**原因**：`onnxruntime/core/providers/cuda/cu_inc/common.cuh` 的 `_IsInf::operator()`
用的是 `if constexpr` 链，四个分支（含最后的 `else`）每个都 `return`，源码本身
没有漏返回（上游 v1.20.1 一字不差）。是 CUDA 11.4 的 nvcc 前端太旧，分支分析
认不出所有路径都返回，误报缺 return。与 3.2.3 的 bf16 同源：ORT 1.18 的代码
超出了 JetPack 5.1.2 自带编译器的能力。

**处理**：在 `if constexpr` 链结束后补一句永远执行不到的 `return false;`，只为哄过
老版前端，不改变任何语义。用 gedit 打开（第 546~559 行是这一段）：

```bash
gedit +552 /mnt/data/code/3rd/onnxruntime-1.18.0/onnxruntime/core/providers/cuda/cu_inc/common.cuh
```

把这段：

```cpp
    } else if constexpr (detect_negative) {
      return isinf_details::IsInfTyped<T>::IsInfNeg(a);
    } else {
      return false;
    }
  }
};
```

改成：

```cpp
    } else if constexpr (detect_negative) {
      return isinf_details::IsInfTyped<T>::IsInfNeg(a);
    } else {
      return false;
    }
    // nvcc 11.4 的返回路径分析认不出 if constexpr 已覆盖所有分支，
    // 显式补一句让它放过；实际不可达，不改语义。
    return false;
  }
};
```

**第二处同类误报：`max_pool_with_index.cu`（62%~63%）**

补完 `common.cuh` 重跑，编译到 62%~63% 时
`onnxruntime/core/providers/cuda/nn/max_pool_with_index.cu` 第 52 行再报 5 个同格式错误：

```
error: missing return statement at end of non-void function
"lambda [](int, int, int, int, int)->int64_t"
  detected during instantiation of MaxPoolWithIndexKernel<T,Layout>
  [with T=float, Layout=false] ...
make[2]: *** [.../max_pool_with_index.cu.o] Error 1
make[1]: *** [.../onnxruntime_providers_cuda.dir/all] Error 2
```

`compute_offset` lambda 里只有 `if constexpr (Layout == LAYOUT_NCHW)` 和
`else if constexpr (Layout == LAYOUT_NHWC)` 两个分支、没有最后的 `else`。`Layout`
是 bool 模板参数，两个分支实际已覆盖全部取值，但 CUDA 11.4 同样认不出，
处理方式和 `common.cuh` 一致——**只是返回值要跟函数返回类型一致，这里是 `int64_t`**：

```bash
gedit +48 /mnt/data/code/3rd/onnxruntime-1.18.0/onnxruntime/core/providers/cuda/nn/max_pool_with_index.cu
```

把第 52~55 行改成：

```cpp
    } else if constexpr (Layout == LAYOUT_NHWC) {
      return (((n_index * height + h_index) * width + w_index) * depth + d_index) * channels + c_index;
    }
    // nvcc 11.4 的返回路径分析认不出 if constexpr 已覆盖所有分支，
    // 显式补一句让它放过；实际不可达，不改语义。
    return 0;
  };
```

**第三处同类误报：`moe_cutlass_kernel.h` 的 `CreateMMA` lambda（CUTLASS / MoE，75% 左右）**

`contrib_ops/cuda/moe/ft_moe/moe_gemm_kernels_fp16_uint4.cu` 会报 36 个错误，真正位置在
`onnxruntime/contrib_ops/cuda/moe/cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h`：

```
moe_cutlass_kernel.h(373): error: missing return statement at end of non-void function
"lambda []()->cutlass::gemm::threadblock::MmaMultistage<...>"
  detected during instantiation of MoeFCGemm<...>::run_kernel_ ...
```

同一个模板被多种配置实例化，所以同一条错误会重复很多遍，「大量报错」其实是复读。

原因还是 nvcc 11.4 认不出 `if constexpr` 分支已经覆盖所有返回路径。但这处**不能照抄前两处**
在末尾补一句 return：

- lambda 的返回类型是整个 CUTLASS `Mma` 模板，写不出一个通用的兜底值；
- 它有两种实例：普通 `MmaMultistage` 用 4 参数构造，DQ 版本 `DqMmaMultistage` 只接受
  带 `group_size` 的 5 参数构造，补在 if/else 后面的任何一句都会在其中一种实例下编不过。

处理方式：把 lambda 换成按 `use_dq_gemm` 分派的两个函数重载，每个函数体只有一条 return，
每种实例只实例化自己需要的构造函数。

第一处，把第 368~374 行的 `CreateMMA` lambda **连同下面的调用一起**替换：

```bash
gedit +368 /mnt/data/code/3rd/onnxruntime-1.18.0/onnxruntime/contrib_ops/cuda/moe/cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h
```

改前：

```cpp
      // Construct thread-scoped matrix multiply
      auto CreateMMA = [&]() {
        if constexpr (use_dq_gemm<Mma>::value)
          return Mma(shared_storage.main_loop, params.group_size, thread_idx, warp_idx, lane_idx);
        else
          return Mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
      };
      Mma mma = CreateMMA();
```

改后：

```cpp
      // Construct thread-scoped matrix multiply
      Mma mma = create_mma<Mma>(shared_storage, params.group_size, thread_idx, warp_idx, lane_idx,
                                use_dq_gemm<Mma>{});
```

第二处，在第 53~54 行 `use_dq_gemm` 的定义之后插入两个重载：

```cpp
// nvcc 11.4 的返回路径分析认不出 if constexpr 已覆盖所有分支，会在 run_kernel_ 里
// 构造 mma 的 lambda 上误报 missing return statement。按 use_dq_gemm 拆成两个只有
// 单条 return 的重载，每种实例只实例化自己需要的构造函数，绕开该误报。
template <typename Mma, typename SharedStorage>
CUTLASS_DEVICE Mma create_mma(SharedStorage& shared_storage, int group_size, int thread_idx,
                              int warp_idx, int lane_idx, platform::true_type) {
  return Mma(shared_storage.main_loop, group_size, thread_idx, warp_idx, lane_idx);
}

template <typename Mma, typename SharedStorage>
CUTLASS_DEVICE Mma create_mma(SharedStorage& shared_storage, int /*group_size*/, int thread_idx,
                              int warp_idx, int lane_idx, platform::false_type) {
  return Mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
}
```

- **替换时务必连 `Mma mma = CreateMMA();` 一起换掉**。只删 lambda 定义、留下旧调用，
  下一轮会报 `identifier "CreateMMA" is undefined`（第一轮实测踩过）。
- 改完直接重跑 3.3.5 的命令，增量编译只重编包含这个头文件的几个 MoE `.cu`
  （`moe_gemm_kernels_fp16_uint4.cu`、`fp16_fp16`、`fp32_fp32` 等），几分钟。
- 这类 `missing return statement` 在 nvcc 里本来是警告 #940-D，被 ORT 的
  `COMPILE_WARNING_AS_ERROR ON`（等价 nvcc `-Werror all-warnings`）升级成了 error。
  不想逐处补的话可以整体加 `--compile_no_warning_as_error`，代价是编译参数变化会全量重编。

- 三处修改都只需做一次；**重新解压 ORT 源码后要重做**，否则下次编译还会报。
- 改完直接重跑 3.3.5 的命令，不用清 `build/Linux/Release`：CMake 沿用已有目标
  文件，只重编改过的 `.cu` 及其后续目标。
- 若其他 `.cu` 再报同样格式的错误，先看返回类型：简单类型按上面的方式在函数末尾补一句
  不可达的 return（`bool` → `return false;`，`int64_t` → `return 0;`）；如果返回类型是
  复杂模板、且不同实例的构造函数不同，就按第三处的办法拆成按标记分派的重载。

#### 3.3.7 CUDA 11.4 的 cicc 生成错误 LLVM IR（`resize_antialias_impl.cu`，用替换文件）

**现象**：继续编译到 `onnxruntime/core/providers/cuda/tensor/resize_antialias_impl.cu`
时，日志先刷出大量：

```
Call parameter type does not match function signature!
  %tmp141 = load %struct._ZSt5tupleIJllEE*, %struct._ZSt5tupleIJllEE** %__T1875, align 8
 call void @_ZN11onnxruntime4cuda37_SetupBilinearUpsampleFilterAntiAliasIf...
<unnamed>: parse Invalid record (Producer: 'LLVM7.0.1' Reader: 'LLVM 7.0.1')
make[2]: *** [.../resize_antialias_impl.cu.o] Error 1
```

**原因**：这不是 `if constexpr` 的返回路径问题，而是 CUDA 11.4 的 nvcc 设备端编译器
cicc（基于 LLVM 7）的已知 bug：`_SetupBilinearUpsampleFilterAntiAlias` 和
`_SetupTrilinerarUpsampleFilterAntiAlias` 两个 `__global__` kernel 按值接收
`std::tuple` 参数，且是在 `DISPATCH_ANTIALIAS_FILTER_SETUP` 宏的 lambda 里启动的，
cicc 在这个组合下生成的 LLVM IR 参数类型不匹配，直接解析失败。CUDA 11.6 / 11.8
配 GCC 10.x 也会复现；官方已不再支持 CUDA 11.x 与 Ubuntu 20.04，不会修。
JetPack 5.1.2 也换不到 CUDA 11.8 / 12。

来源：microsoft/onnxruntime issue
[#20330 "Build on Linux with CUDA support"](https://github.com/microsoft/onnxruntime/issues/20330)。
该 issue 里多位 Jetson 用户复现了同样的报错，维护者确认是 nvcc/cicc 的问题且不再修复；
issue 后半段由 OlegZhavoronkov 给出可用的手工改法——把宏内的 kernel 启动语句挪进
独立的 `__host__` 包装函数。3.3.7 的替换文件就是按这个思路整理出的 1.18.0 完整版本。

**处理**：不用手工改源码。项目 `onnxruntime/install/onnxruntime-1.18.0/` 下已放好
改过的 `resize_antialias_impl.cu`（1.18.0 原文件 + 两个 `__host__` 包装函数 + 三处
调用点改名，纯编译期改写、语义不变），直接覆盖源码里的同名文件：

```bash
cp <项目根>/onnxruntime/install/onnxruntime-1.18.0/resize_antialias_impl.cu \
   /mnt/data/code/3rd/onnxruntime-1.18.0/onnxruntime/core/providers/cuda/tensor/   
```
或者在onnxruntime根目录
```bash
cp /install/onnxruntime-1.18.0/resize_antialias_impl.cu \
   /mnt/data/code/3rd/onnxruntime-1.18.0/onnxruntime/core/providers/cuda/tensor/   
```

- 覆盖后直接重跑 3.3.5 的命令，增量编译只重编这个 `.cu`，不用清 `build/Linux/Release`。
- `install/` 不入库，这个替换文件只在本地；**换机器或重新拉仓库时要重新放进
  `onnxruntime/install/onnxruntime-1.18.0/`**，重新解压 ORT 源码后也要重新 `cp`。

### 3.4 安装到本目录

在 Jetson 上、且项目仓库已同步到设备后执行（把 `<项目根>` 换成实际路径，
例如 `/home/nvidia/audio_ai_process`）：

```bash
ORT_SRC="/mnt/data/code/3rd/onnxruntime-1.18.0"
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
ORT_SRC="/mnt/data/code/3rd/onnxruntime-1.18.0"
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
| CMake 报版本过低 | JetPack 自带 3.16，按 3.3.1 在 conda 环境装 3.28.3；`which cmake` 必须指向 env 的 `bin` |
| 装了 cmake 仍显示 3.16.3 | 先 `hash -r` 清 bash 的命令缓存；仍不对再检查 conda 环境是否激活、`which cmake` 是否指向 env 的 `bin` |
| pip 装了 cmake 仍报 `Could not find CMAKE_ROOT` | 多半是 shell 缓存仍指向旧的 conda cmake（其模块目录已缺失）；`hash -r` 后重试，仍不行再新建干净环境 |
| 链接阶段被杀（OOM） | 增大 swap，`--parallel` 降到 4 |
| 编译报 `missing return statement at end of non-void function`（`_IsInf`、`max_pool_with_index.cu` 的 lambda、或 CUTLASS/MoE 的 `moe_cutlass_kernel.h`） | CUDA 11.4 的 nvcc 前端对 `if constexpr` 误报，不是代码问题；按 3.3.6 处理：简单返回类型补一句不可达的 return（`bool` → `return false;`，`int64_t` → `return 0;`），复杂模板按 `moe_cutlass_kernel.h` 那样拆成按标记分派的重载 |
| 编译刷大量 `Call parameter type does not match function signature!`，最后 `parse Invalid record (Producer: 'LLVM7.0.1')`（`resize_antialias_impl.cu`） | CUDA 11.4 的 cicc 在宏内启动带 `std::tuple` 参数的 kernel 时生成错误 LLVM IR，不是代码问题；按 3.3.7 用 `install/onnxruntime-1.18.0/resize_antialias_impl.cu` 覆盖源码同名文件后重跑 |
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
| 2026-11-03 | 按 Jetson 实测重写 3.2 前置条件；确认 C++ 共享库构建不要求任何 Python 库，复用已有 conda 环境与 `/mnt/data` 构建路径；源码改用官方 tarball + `--skip_submodule_sync` |
| 2026-11-03 | 编译首次实测：补 `tee` 日志（tmux 仅 SSH 需要，NoMachine 桌面不需要）；确认 v1.18.0 `build.py` 无 `--install`，产物在 `build/Linux/Release/`，安装见 3.4 |
| 2026-11-03 | 依赖下载实测 date v3.0.1 被 GitHub 掐断（curl 56）；新增 `fetch_ort_deps.py`（镜像下载 + SHA1 校验 + 改写 deps.txt 为本地路径），依赖预先下载到 `install/onnxruntime-1.18.0/ort_deps/` |
| 2026-11-03 | 章节固定为 3.3.4 预下载依赖 / 3.3.5 编译：Jetson 现场下载失败，改为开发机先下好、随项目同步后在 Jetson 上重跑脚本校验并改写路径；3.3.4 补充「为什么必须改 deps.txt」与跨机同步的路径说明 |
| 2026-11-03 | 实测 eigen 归档 SHA1 过期（GitLab 重新打包，同 commit 内容与 GitHub 镜像零差异）；脚本增加 `SHA1_OVERRIDES`，将 eigen 期望值改为 `32b145f5...` 并写入改写后的 deps.txt |
| 2026-11-03 | 编译实测配置阶段失败：ORT 1.17 起对 aarch64 强制要求 `-march=armv8.2-a+bf16`（GCC ≥ 10），JetPack 5.1.2 自带 GCC 9.4 不满足；新增 3.2.3 编译工具小节，3.2 前置检查命令与实测表补充 nvcc 上限、libstdc++、gcc-10 可装性；硬门槛由三条改为四条 |
| 2026-11-03 | 3.3.5 编译命令改用 gcc-10（`CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` / `CMAKE_CUDA_HOST_COMPILER` 三个定义），补充换编译器必须清空 `build/Linux/Release` 与配置阶段的验收点 |
| 2026-11-03 | 编译到 53% 实测 CUDA 11.4 的 nvcc 误报 `missing return statement`（`common.cuh` 的 `_IsInf` 用了 `if constexpr` 链，源码无误、上游 1.20.1 相同）；新增 3.3.6 记录现象、原因与 gedit 补一行 `return false;` 的处理，3.6 常见问题补对应条目 |
| 2026-11-03 | 编译到 62%~63% 实测第二处同格式误报（`max_pool_with_index.cu` 的 `int64_t` lambda），3.3.6 补充该处处理并说明补的返回值必须与函数类型一致；随后 58% 回退重编时实测 `resize_antialias_impl.cu` 触发 cicc 的 LLVM IR 错误，新增 3.3.7，替换文件放 `install/onnxruntime-1.18.0/`，3.6 补对应条目 |
| 2026-11-03 | 实测 84%~86% 阶段仍会编译单元测试目标（`--skip_tests` 只跳过运行），3.3.5 编译命令增加 `onnxruntime_BUILD_UNIT_TESTS=OFF` 并在参数表说明两者区别；补充 GCC 10.1 的 `std::pair` 传参 ABI note 不是报错、可忽略 |
| 2026-11-03 | 编译到 75% 左右实测第三处 `missing return statement`（CUTLASS `moe_cutlass_kernel.h` 的 `CreateMMA` lambda，36 个错误为同一模板的多次实例化）；返回类型是复杂模板且 DQ/非 DQ 构造函数不同，改为 `create_mma` 标记分派重载，3.3.6 补充该小节与「漏改调用会报 CreateMMA undefined」的坑 |
| 2026-11-03 | **1.18.0 在 Jetson Orin NX 上完整编译通过**：3.3.5 命令（gcc-10 + `onnxruntime_BUILD_UNIT_TESTS=OFF`）配合 3.3.6 三处源码修改与 3.3.7 替换文件，四类 nvcc 问题即本次实测的全部源码级障碍；3.1 补充选 1.18.0 的困难与优势对比 |
| 2026-11-03 | **编译产物与 CUDA EP 真机验证**：`build/Linux/Release/` 下为 `libonnxruntime.so.1.18.0`（20 MB，核心库）、`libonnxruntime_providers_cuda.so`（CUDA kernel 所在）、`libonnxruntime_providers_shared.so`，三者安装时必须同目录；`onnxruntime_test_all --gtest_filter='*Cuda*:*CUDA*'` 在 3.2 秒内跑完 29 个用例并全部 PASSED，另有 2 个源码自带 `DISABLED_` 用例未跑。日志中的 `transformer_memcpy.cc` 2 个 Memcpy 节点提示、`SimplifiedLayerNormalization` 的 `MergeShapeInfo` / `VerifyOutputSizes` 形状不一致（{4,20,1,16} → {4,20,1,1}）都是上游测试模型的 warning，ORT 自动退回宽松合并，不影响数值结果 |
| 2026-11-03 | 新评估 1.17.0 备选路线：逐文件对照源码确认 1.18.0 的四处补丁在 1.17.0 中都不需要（无 `IsInfTyped`、`max_pool_with_index.cu` 无布局分派 lambda、MoE 用 ft_moe 直接构造 `Mma`、无 CUDA antialias 实现）；限制（无 CUDA antialias、仍要 gcc-10、依赖清单不同）与切换成本写入 3.1，主线仍保持 1.18.0；同时补两版 CUDA 口径对照：1.18.0 有 `CMAKE_CUDA_COMPILER_VERSION < 11.4` 的 `FATAL_ERROR`，1.17.0 只有 `< 11.6` 关 flash attention，两者官方 CI/打包镜像均为 CUDA 11.8.0 |
