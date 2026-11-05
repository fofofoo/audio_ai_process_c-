##################################################################################
#  音频 AI 处理工程 —— 两个零耦合库 + 可执行入口
#
#  库：
#    lib_hk_sdk_audio_client.so  麦克风采集：登录、语音转发、G.722.1 解码、
#                               可选降采样、wav 录制、取流缓冲与统计
#    lib_audio_ai_process.so     ASR 处理：LogFBank 前端、AsrDecoder / KwsDecoder
#                               （ORT）、KWS 关键词前缀树、语谱图与文本渲染
#
#  两个库互不 include 对方头文件；组合关系只出现在可执行入口里。
#
#  构建：
#    cmake -S . -B build && cmake --build build -j
#  安装：
#    cmake --install build     # 产物在 <工程根>/install/<库名>/{include,lib}
##################################################################################

cmake_minimum_required(VERSION 3.10)
project(audio_ai_process LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# ---------------------------------------------------------------------
# 平台检测：按架构选择 HCNetSDK 库目录（SDK 自带三套预编译库）
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
# OpenCV：ASR 库的语谱图 / 文本渲染，以及入口里的 imshow / waitKey
# ---------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# ---------------------------------------------------------------------
# HCNetSDK 与采集库路径
# ---------------------------------------------------------------------
set(HK_MIC_DIR    "${CMAKE_SOURCE_DIR}/hk_mic_client")
set(HIK_SDK_ROOT  "${HK_MIC_DIR}/hk_sdk")
set(HIK_INCLUDE_DIRS "${HIK_SDK_ROOT}/include")
set(HIK_LIB_ROOT  "${HIK_SDK_ROOT}/lib/${HIK_LIB_SUBDIR}")

# 音频链路只需要这三个库；其余 .so 是它们的运行期依赖，通过 RPATH 找到
set(HIK_LIBS
    ${HIK_LIB_ROOT}/libhcnetsdk${HIK_LIB_SUFFIX}
    ${HIK_LIB_ROOT}/libHCCore${HIK_LIB_SUFFIX}
    ${HIK_LIB_ROOT}/libPlayCtrl${HIK_LIB_SUFFIX}
)

message(STATUS "HIK SDK root : ${HIK_SDK_ROOT}")
message(STATUS "HIK SDK libs : ${HIK_LIB_ROOT}")

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------
# ONNX Runtime：按架构选择项目内托管的版本
#
#   x86_64  (WSL / PC)       -> onnxruntime-1.23.2
#   aarch64 (Jetson Orin NX) -> onnxruntime-1.18.0
#
# 找不到时只跳过依赖 ORT 的目标（asr_decode.cpp / kws_decode.cpp 与四个
# ORT 入口），音频采集、语谱图与 fbank 对拍仍可构建。
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
    set(ORT_FOUND FALSE)
    message(WARNING
        "ONNX Runtime 未安装：${ORT_LIBRARY}\n"
        "  x86_64  : 确认预编译包已复制到 onnxruntime/install/onnxruntime-1.23.2/\n"
        "  aarch64 : 按 onnxruntime/README.md 第 3 节在 Jetson 上编译并安装")
endif()

# ---------------------------------------------------------------------
# FreeType：text_renderer.cpp 用系统字体渲染中文（cv::putText 只支持 ASCII）。
# 找不到时 text_renderer 编译成空实现，入口靠 ready() 回退 ASCII。
# 注意 OpenCV 可能来自 /usr/local（自带 build 无 freetype 模块），
# 因此不混链发行版 libopencv_freetype。
# ---------------------------------------------------------------------
find_package(Freetype QUIET)
if(FREETYPE_FOUND)
    message(STATUS "FreeType found: 中文识别结果用系统字体渲染")
else()
    message(STATUS "FreeType 未找到：识别结果窗口文字回退 ASCII")
endif()

# =====================================================================
#  一、麦克风采集库（不依赖 OpenCV / ORT）
#
#  登录、语音转发、G.722.1 解码、可选降采样、wav 录制、取流缓冲与统计。
# =====================================================================
set(hk_sdk_audio_client_SOURCES
    ${HK_MIC_DIR}/hk_sdk_audio_client.cpp
    ${HK_MIC_DIR}/audio_resampler.cpp
)

# LIB_DEMOS 每组只放一个库：生成与链接各自独立，不靠分支区分。
set(LIB_DEMOS
    hk_sdk_audio_client
)

# 为每一个动态库链接库文件
foreach(TARGET ${LIB_DEMOS})
    add_library(${TARGET} SHARED ${${TARGET}_SOURCES})

    # 统一输出 build/lib/lib_xxx.so；安装后靠 $ORIGIN 找同目录依赖
    set_target_properties(${TARGET} PROPERTIES
        OUTPUT_NAME "_${TARGET}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        INSTALL_RPATH "$ORIGIN"
    )

    # 公共头文件直接 include "HCNetSDK.h"，
    # 因此 SDK 头文件路径必须对使用方可见；SDK 库本身只在本库内部链接。
    target_include_directories(${TARGET} PUBLIC
        ${HK_MIC_DIR}
        ${HIK_INCLUDE_DIRS}
    )

    # ${TARGET}链接到 HCNetSDK 的库
    target_link_libraries(${TARGET} PRIVATE ${HIK_LIBS})

    # 链接 Threads 库（头文件依赖）
    target_link_libraries(${TARGET} PRIVATE Threads::Threads)

    set_target_properties(${TARGET} PROPERTIES
        BUILD_RPATH "${HIK_LIB_ROOT}"
    )
endforeach()

# =====================================================================
#  二、ASR 处理库（不依赖 HCNetSDK / 采集库）
#
#  LogFBank 前端、AsrDecoder / KwsDecoder（ORT）、KWS 关键词前缀树、
#  语谱图与文本渲染。ORT 缺失时跳过 asr_decode.cpp / kws_decode.cpp 与四个
#  ORT 入口，音频采集、语谱图与 fbank 对拍仍可构建。
# =====================================================================
set(AAP_DIR "${CMAKE_SOURCE_DIR}/audio_ai_process")

set(audio_ai_process_SOURCES
    ${AAP_DIR}/audio_preprocess.cpp
    ${AAP_DIR}/kws_context_graph.cpp
    ${AAP_DIR}/spectrogram_image.cpp
    ${AAP_DIR}/text_utils.cpp
    ${AAP_DIR}/text_renderer.cpp
)
if(ORT_FOUND)
    list(APPEND audio_ai_process_SOURCES
        ${AAP_DIR}/asr_decode.cpp
        ${AAP_DIR}/asr_session.cpp
        ${AAP_DIR}/kws_asr_session.cpp
        ${AAP_DIR}/kws_decode.cpp
    )
endif()

set(LIB_DEMOS
    audio_ai_process
)

# 为每一个动态库链接库文件
foreach(TARGET ${LIB_DEMOS})
    add_library(${TARGET} SHARED ${${TARGET}_SOURCES})

    # 统一输出 build/lib/lib_xxx.so；安装后靠 $ORIGIN 找同目录依赖
    set_target_properties(${TARGET} PROPERTIES
        OUTPUT_NAME "_${TARGET}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        INSTALL_RPATH "$ORIGIN"
    )

    # 公共头文件使用 audio_ai_process/xxx.h 与 opencv2/... 写法，
    # 因此把工作区根目录与 OpenCV 头文件路径传给使用方。
    target_include_directories(${TARGET} PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    )

    # ${TARGET}链接到 OpenCV 的库
    target_link_libraries(${TARGET} PUBLIC ${OpenCV_LIBS})

    # 链接 Threads 库（头文件依赖）
    target_link_libraries(${TARGET} PUBLIC Threads::Threads)

    if(FREETYPE_FOUND)
        # ${TARGET}链接到 FreeType 的库（中文文本渲染）
        target_compile_definitions(${TARGET} PRIVATE HK_HAVE_FREETYPE=1)
        target_include_directories(${TARGET} PRIVATE ${FREETYPE_INCLUDE_DIRS})
        target_link_libraries(${TARGET} PRIVATE ${FREETYPE_LIBRARIES})
    endif()

    if(ORT_FOUND)
        # ${TARGET}链接到 ONNX Runtime 的库
        # providers_cuda.so 由 ORT 运行时 dlopen，必须与
        # libonnxruntime.so 同目录，因此 BUILD_RPATH 指向 ORT_LIB_DIR。
        target_include_directories(${TARGET} PUBLIC ${ORT_INCLUDE_DIR})
        target_link_libraries(${TARGET} PUBLIC ${ORT_LIBRARY})

        set_target_properties(${TARGET} PROPERTIES
            BUILD_RPATH "${ORT_LIB_DIR}"
        )
    endif()
endforeach()

# =====================================================================
#  三、可执行入口
#
#  入口只做「参数定义 + 数据源 + 装配」，库负责流水线；
# 每个入口只链接自己需要的库。
# =====================================================================

# --- 3.1 声明入口 ---
add_executable(save_audio_wav.out save_audio_wav.cpp)
add_executable(test_spectrogram_cv_mic.out test_spectrogram_cv_mic.cpp)
add_executable(test_spectrogram_cv_wav.out test_spectrogram_cv_wav.cpp)
add_executable(fbank_dump.out tools/fbank_dump.cpp)
add_executable(test_context_graph.out tools/test_context_graph.cpp)

# 依赖 ORT 的入口：未安装 ONNX Runtime 时整体跳过
if(ORT_FOUND)
    add_executable(test_wav_asr.out test_wav_asr.cpp)
    add_executable(test_wav_kws.out test_wav_kws.cpp)
    add_executable(test_mic_asr.out test_mic_asr.cpp)
    add_executable(test_mic_kws_asr.out test_mic_kws_asr.cpp)
    add_executable(load_onnx_model.out load_onnx_model.cpp)
endif()

# --- 3.2 按依赖分组，用 foreach 统一设置 ---
# 每个入口只链接自己需要的库：
#   仅采集库          save_audio_wav
#   仅 ASR 库         test_spectrogram_cv_wav / fbank_dump / test_wav_asr /
#                     test_wav_kws / load_onnx_model
#   两个库都链接      test_spectrogram_cv_mic / test_mic_asr
set(EXEC_DEMOS
    save_audio_wav.out
    test_spectrogram_cv_mic.out
    test_spectrogram_cv_wav.out
    fbank_dump.out
    test_context_graph.out
)

set(EXEC_LINK_HK
    save_audio_wav.out
    test_spectrogram_cv_mic.out
)

set(EXEC_LINK_AAP
    test_spectrogram_cv_mic.out
    test_spectrogram_cv_wav.out
    fbank_dump.out
    test_context_graph.out
)

if(ORT_FOUND)
    list(APPEND EXEC_DEMOS
        test_wav_asr.out test_wav_kws.out test_mic_asr.out
        test_mic_kws_asr.out load_onnx_model.out)
    list(APPEND EXEC_LINK_HK test_mic_asr.out test_mic_kws_asr.out)
    list(APPEND EXEC_LINK_AAP
        test_wav_asr.out test_wav_kws.out test_mic_asr.out
        test_mic_kws_asr.out load_onnx_model.out)
endif()

foreach(TARGET ${EXEC_DEMOS})
    # 入口用 HK_PROJECT_DIR 拼 weight/、results/ 的绝对路径
    target_compile_definitions(${TARGET} PRIVATE
        HK_PROJECT_DIR="${CMAKE_SOURCE_DIR}"
    )

    # ${TARGET} 链接到 OpenCV 的库
    # audio_ai_process 已用 PUBLIC 传递，这里再显式写出：
    # 每个入口依赖了什么一眼可见，也便于日后单独替换某个入口的链接。
    target_link_libraries(${TARGET} PRIVATE ${OpenCV_LIBS})
    target_include_directories(${TARGET} PRIVATE ${OpenCV_INCLUDE_DIRS})

    # 链接 Threads 库（头文件依赖）
    target_link_libraries(${TARGET} PRIVATE Threads::Threads)

    # ${TARGET} 链接到麦克风采集库（设备登录 / G.722.1 解码 / 降采样）
    # 只给真正从设备取流的入口；纯 wav 与前端工具不链接，
    # 避免无谓加载 HCNetSDK，也不把 SDK 运行库带进不需要它的进程。
    if(${TARGET} IN_LIST EXEC_LINK_HK)
        target_link_libraries(${TARGET} PRIVATE hk_sdk_audio_client)

        # ${TARGET} 链接到 HCNetSDK 的库
        # 采集库已 PRIVATE 链接，这里显式写出，保持每个入口依赖可见；
        # 头文件走 PUBLIC 传递，但入口直接 include HCNetSDK.h 时也能找到。
        target_link_libraries(${TARGET} PRIVATE ${HIK_LIBS})
        target_include_directories(${TARGET} PRIVATE ${HIK_INCLUDE_DIRS})

        # 用 APPEND：test_mic_asr 同时需要 HCNetSDK 与 ORT 两个搜索路径
        set_property(TARGET ${TARGET} APPEND PROPERTY
            BUILD_RPATH "${HIK_LIB_ROOT}")
        set_property(TARGET ${TARGET} APPEND PROPERTY
            INSTALL_RPATH "${HIK_LIB_ROOT}")
    endif()

    # ${TARGET} 链接到 ASR 处理库（LogFBank / 语谱图 / 解码 / 中文渲染）
    if(${TARGET} IN_LIST EXEC_LINK_AAP)
        target_link_libraries(${TARGET} PRIVATE audio_ai_process)

        if(ORT_FOUND)
            # ${TARGET} 链接到 ONNX Runtime 的库
            # 处理库内部已依赖 ORT，这里显式列出便于查看；
            # load_onnx_model 直接调用 ORT 接口，也由这段统一覆盖。
            target_link_libraries(${TARGET} PRIVATE ${ORT_LIBRARY})
            target_include_directories(${TARGET} PRIVATE ${ORT_INCLUDE_DIR})

            set_property(TARGET ${TARGET} APPEND PROPERTY
                BUILD_RPATH "${ORT_LIB_DIR}")
            set_property(TARGET ${TARGET} APPEND PROPERTY
                INSTALL_RPATH "${ORT_LIB_DIR}")
        endif()
    endif()
endforeach()

# =====================================================================
#  四、安装：install/<库名>/{include,lib}
# =====================================================================

# 头文件显式列出，避免 GLOB 把 hk_sdk/include 下的厂商头文件也装进来。
set(HK_AUDIO_HEADERS
    ${HK_MIC_DIR}/hk_sdk_audio_client.h
    ${HK_MIC_DIR}/audio_resampler.h
)

set(AAP_HEADERS
    ${AAP_DIR}/audio_preprocess.h
    ${AAP_DIR}/block_worker.h
    ${AAP_DIR}/kws_context_graph.h
    ${AAP_DIR}/spectrogram_image.h
    ${AAP_DIR}/text_utils.h
    ${AAP_DIR}/text_renderer.h
)
if(ORT_FOUND)
    list(APPEND AAP_HEADERS
        ${AAP_DIR}/asr_decode.h
        ${AAP_DIR}/asr_session.h
        ${AAP_DIR}/kws_asr_session.h
        ${AAP_DIR}/kws_decode.h
    )
endif()

# 安装一个模块：头文件 + 自身共享库。
# HEADER_DEST 为空表示头文件平铺到 include/；否则放到 include/<HEADER_DEST>/。
function(install_module TARGET_NAME MODULE_INSTALL_ROOT HEADER_FILES HEADER_DEST)
    if(NOT TARGET ${TARGET_NAME})
        message(FATAL_ERROR "install_module: target '${TARGET_NAME}' does not exist.")
    endif()

    if(HEADER_FILES)
        if(HEADER_DEST)
            install(FILES ${HEADER_FILES}
                DESTINATION "${MODULE_INSTALL_ROOT}/include/${HEADER_DEST}")
        else()
            install(FILES ${HEADER_FILES}
                DESTINATION "${MODULE_INSTALL_ROOT}/include")
        endif()
    endif()

    install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION "${MODULE_INSTALL_ROOT}/lib"
        ARCHIVE DESTINATION "${MODULE_INSTALL_ROOT}/lib"
        RUNTIME DESTINATION "${MODULE_INSTALL_ROOT}/bin"
    )
endfunction()

set(INSTALL_ROOT "${CMAKE_SOURCE_DIR}/install")

# 采集库：头文件平铺；SDK 的 .so 与 HCNetSDKCom/ 一并装到 lib，
# 库自身 INSTALL_RPATH=$ORIGIN，运行时能在同目录找到 libhcnetsdk。
install_module(hk_sdk_audio_client
    "${INSTALL_ROOT}/hk_sdk_audio_client"
    "${HK_AUDIO_HEADERS}"
    ""
)
# 公共头文件 hk_sdk_audio_client.h 直接 include "HCNetSDK.h"，
# 因此 SDK 头文件也必须随库安装，否则使用方无法编译。
install(DIRECTORY "${HIK_INCLUDE_DIRS}/"
    DESTINATION "${INSTALL_ROOT}/hk_sdk_audio_client/include"
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
)
install(DIRECTORY "${HIK_LIB_ROOT}/"
    DESTINATION "${INSTALL_ROOT}/hk_sdk_audio_client/lib"
    FILES_MATCHING
    PATTERN "*.so*"
    PATTERN "*.dll"
)

# ASR 库：头文件保留 audio_ai_process/ 前缀，使用方 include 路径与源码一致。
# ORT / OpenCV 不复制，属于外部运行期依赖（见 README）。
install_module(audio_ai_process
    "${INSTALL_ROOT}/audio_ai_process"
    "${AAP_HEADERS}"
    "audio_ai_process"
)
