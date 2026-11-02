##################################################################################
#  海康设备端音频录制 demo  ——  最小可构建配置
#
#  目标：
#    save_audio_wav.out           录制固定时长设备音频并保存为 wav
#    test_spectrogram_cv_mic.out  设备实时音频 -> LogFBank -> OpenCV 语谱图
#    test_spectrogram_cv_wav.out  wav 文件    -> LogFBank -> OpenCV 语谱图
#
#  只依赖：
#    - hk_mic_client/hk_sdk_audio_client.{h,cpp}   音频采集客户端
#    - hk_mic_client/hk_sdk/                       HCNetSDK 头文件与库
#
#  构建：
#    cmake -S . -B build && cmake --build build -j
#  运行：
#    ./build/save_audio_wav.out        # wav 写入 <工程根>/results/
##################################################################################

cmake_minimum_required(VERSION 3.10)
project(hk_audio_record LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# ---------------------------------------------------------------------
# 平台检测：按架构选择 SDK 库目录（SDK 自带三套预编译库）
# ---------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(HIK_LIB_SUBDIR "linux64_arm")
        set(HIK_LIB_SUFFIX ".so")
        message(STATUS "Detected Platform: Linux ARM64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
        set(HIK_LIB_SUBDIR "linux_x86")
        set(HIK_LIB_SUFFIX ".so")
        message(STATUS "Detected Platform: Linux x86_64")
    else()
        message(FATAL_ERROR "Unsupported Linux architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(HIK_LIB_SUBDIR "win64_x86")
        set(HIK_LIB_SUFFIX ".lib")
        message(STATUS "Detected Platform: Windows x64")
    else()
        message(FATAL_ERROR "Unsupported Windows architecture (Expected 64-bit)")
    endif()
else()
    message(FATAL_ERROR "Unsupported OS: ${CMAKE_SYSTEM_NAME}")
endif()

# ---------------------------------------------------------------------
# ONNX Runtime：按架构选择项目内托管的版本
#
#   x86_64  (WSL / PC)       -> onnxruntime-1.23.2（官方预编译，CUDA 12.x + cuDNN 9.x）
#   aarch64 (Jetson Orin NX) -> onnxruntime-1.18.0（按 onnxruntime/README.md 第 3 节源码编译）
#
# 两个目录结构统一为 <root>/include 与 <root>/lib，CMake 只认这套路径。
# 需要推理能力的目标调用 ort_link_target(<target>)，一次性设置 include、链接和 RPATH。
# ---------------------------------------------------------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ORT_VERSION_DIR "onnxruntime-1.18.0")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(ORT_VERSION_DIR "onnxruntime-1.23.2")
else()
    message(FATAL_ERROR "ONNX Runtime: unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(ORT_ROOT        "${CMAKE_SOURCE_DIR}/onnxruntime/install/${ORT_VERSION_DIR}")
set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
set(ORT_LIB_DIR     "${ORT_ROOT}/lib")
set(ORT_LIBRARY     "${ORT_LIB_DIR}/libonnxruntime.so")

if(EXISTS "${ORT_LIBRARY}")
    set(ORT_FOUND TRUE)
    message(STATUS "ONNX Runtime root : ${ORT_ROOT}")
else()
    # 不中断现有构建：当前工程还没有推理目标时，ORT 缺失不影响音频链路。
    set(ORT_FOUND FALSE)
    message(WARNING
        "ONNX Runtime 未安装：${ORT_LIBRARY}\n"
        "  x86_64  : 确认预编译包已复制到 onnxruntime/install/onnxruntime-1.23.2/\n"
        "  aarch64 : 按 onnxruntime/README.md 第 3 节在 Jetson 上编译并安装")
endif()

# 把 ORT 接入指定 target：头文件、共享库、运行期 RPATH。
# providers_cuda.so 由 ORT 运行时 dlopen，必须与 libonnxruntime.so 同目录。
function(ort_link_target target)
    if(NOT ORT_FOUND)
        message(FATAL_ERROR "ort_link_target(${target}) 失败：ONNX Runtime 未安装")
    endif()
    target_include_directories(${target} PRIVATE ${ORT_INCLUDE_DIR})
    target_link_libraries(${target} PRIVATE ${ORT_LIBRARY})
    set_target_properties(${target} PROPERTIES
        BUILD_RPATH   "${ORT_LIB_DIR}"
        INSTALL_RPATH "${ORT_LIB_DIR}"
    )
endfunction()

# ---------------------------------------------------------------------
# SDK 与客户端路径
# ---------------------------------------------------------------------
set(HK_MIC_DIR "${CMAKE_SOURCE_DIR}/hk_mic_client")
set(HIK_SDK_ROOT "${HK_MIC_DIR}/hk_sdk")
set(HIK_INCLUDE_DIRS "${HIK_SDK_ROOT}/include")
set(HIK_LIB_ROOT "${HIK_SDK_ROOT}/lib/${HIK_LIB_SUBDIR}")

# 音频链路只需要这三个库；其余 .so 是它们的运行期依赖，通过 RPATH 找到
set(HIK_LIBS
    ${HIK_LIB_ROOT}/libhcnetsdk${HIK_LIB_SUFFIX}
    ${HIK_LIB_ROOT}/libHCCore${HIK_LIB_SUFFIX}
    ${HIK_LIB_ROOT}/libPlayCtrl${HIK_LIB_SUFFIX}
)

message(STATUS "HIK SDK root : ${HIK_SDK_ROOT}")
message(STATUS "HIK SDK libs : ${HIK_LIB_ROOT}")

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# ---------------------------------------------------------------------
# 可执行文件：save_audio_wav.out
# ---------------------------------------------------------------------
add_executable(save_audio_wav.out
    save_audio_wav.cpp
    ${HK_MIC_DIR}/hk_sdk_audio_client.cpp
)

# 注入工程根目录：main 用它把 wav 默认写到 <工程根>/results/
target_compile_definitions(save_audio_wav.out PRIVATE
    HK_PROJECT_DIR="${CMAKE_SOURCE_DIR}"
)

target_include_directories(save_audio_wav.out PRIVATE
    ${HK_MIC_DIR}
    ${HIK_INCLUDE_DIRS}
)

target_link_libraries(save_audio_wav.out PRIVATE
    ${HIK_LIBS}
    Threads::Threads
)

# 运行期在 SDK 库目录里找 libhcnetsdk 及其依赖（HCNetSDKCom 等）
set_target_properties(save_audio_wav.out PROPERTIES
    BUILD_RPATH "${HIK_LIB_ROOT}"
    INSTALL_RPATH "${HIK_LIB_ROOT}"
)

# ---------------------------------------------------------------------
# 可执行文件：test_spectrogram_cv_mic.out
#   设备音频 -> get_last_audio -> AudioPreprocess -> SpectrogramImage -> imshow
# ---------------------------------------------------------------------
add_executable(test_spectrogram_cv_mic.out
    test_spectrogram_cv_mic.cpp
    audio_ai_process/audio_preprocess.cpp
    audio_ai_process/spectrogram_image.cpp
    ${HK_MIC_DIR}/hk_sdk_audio_client.cpp
)

target_include_directories(test_spectrogram_cv_mic.out PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${HK_MIC_DIR}
    ${HIK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_spectrogram_cv_mic.out PRIVATE
    ${OpenCV_LIBS}
    ${HIK_LIBS}
    Threads::Threads
)

set_target_properties(test_spectrogram_cv_mic.out PROPERTIES
    BUILD_RPATH "${HIK_LIB_ROOT}"
    INSTALL_RPATH "${HIK_LIB_ROOT}"
)

# ---------------------------------------------------------------------
# 可执行文件：test_spectrogram_cv_wav.out
#   wav 文件 -> AudioPreprocess -> SpectrogramImage -> imshow
#   不依赖海康 SDK；wav 解析写在本入口的匿名 namespace 里。
# ---------------------------------------------------------------------
add_executable(test_spectrogram_cv_wav.out
    test_spectrogram_cv_wav.cpp
    audio_ai_process/audio_preprocess.cpp
    audio_ai_process/spectrogram_image.cpp
    audio_ai_process/text_utils.cpp
)

target_include_directories(test_spectrogram_cv_wav.out PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_spectrogram_cv_wav.out PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)

# ---------------------------------------------------------------------
# 可执行文件：fbank_dump.out
#   前端对拍工具：raw float32 PCM -> AudioPreprocess -> raw float32 特征
#   由 tools/compare_fbank.py 调用，与 Python 参考实现逐点比较。
# ---------------------------------------------------------------------
add_executable(fbank_dump.out
    tools/fbank_dump.cpp
    audio_ai_process/audio_preprocess.cpp
)

target_include_directories(fbank_dump.out PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(fbank_dump.out PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)

# ---------------------------------------------------------------------
# 可执行文件：test_wav_asr.out
#   wav 文件 -> AudioPreprocess -> AsrDecoder（ORT 三模型）-> 文本
#   可选 SpectrogramImage 窗口 + 底部识别文字；依赖 OpenCV 与 ONNX Runtime。
# ---------------------------------------------------------------------
add_executable(test_wav_asr.out
    test_wav_asr.cpp
    audio_ai_process/audio_preprocess.cpp
    audio_ai_process/asr_decode.cpp
    audio_ai_process/text_utils.cpp
    audio_ai_process/text_renderer.cpp
    audio_ai_process/spectrogram_image.cpp
)

target_compile_definitions(test_wav_asr.out PRIVATE
    HK_PROJECT_DIR="${CMAKE_SOURCE_DIR}"
)

target_include_directories(test_wav_asr.out PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_wav_asr.out PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)

# FreeType：text_renderer.cpp 按 UTF-8 渲染中文识别结果（cv::putText 只支持 ASCII）。
# 直接用系统 FreeType，不依赖 opencv_contrib；找不到时 text_renderer 编译成空实现，
# 入口靠 ready() 回退 ASCII，不影响构建与识别。注意 OpenCV 可能来自 /usr/local
# （自带 build 无 freetype 模块），因此不混链发行版 libopencv_freetype。
find_package(Freetype QUIET)
if(FREETYPE_FOUND)
    target_compile_definitions(test_wav_asr.out PRIVATE HK_HAVE_FREETYPE=1)
    target_include_directories(test_wav_asr.out PRIVATE ${FREETYPE_INCLUDE_DIRS})
    target_link_libraries(test_wav_asr.out PRIVATE ${FREETYPE_LIBRARIES})
else()
    message(STATUS "FreeType 未找到：test_wav_asr 窗口文字回退 ASCII")
endif()

ort_link_target(test_wav_asr.out)

# ---------------------------------------------------------------------
# 可执行文件：load_onnx_model.out
#   模型加载自检：打开 encoder / decoder / joiner，打印输入输出与 metadata。
#   不依赖 OpenCV 和海康 SDK，只依赖 ONNX Runtime。
# ---------------------------------------------------------------------
add_executable(load_onnx_model.out
    load_onnx_model.cpp
    audio_ai_process/text_utils.cpp
)

# 默认模型目录按工程根拼接，入口从任意工作目录启动都能找到 weight/
target_compile_definitions(load_onnx_model.out PRIVATE
    HK_PROJECT_DIR="${CMAKE_SOURCE_DIR}"
)

target_include_directories(load_onnx_model.out PRIVATE ${CMAKE_SOURCE_DIR})

ort_link_target(load_onnx_model.out)
