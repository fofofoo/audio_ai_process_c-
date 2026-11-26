##################################################################################
#  hk_mic_client —— 海康拾音器音频采集 + 特征工程库
#
#  职责：
#    lib_hk_mic_client.so  登录、语音转发、G.722.1 解码、可选降采样、wav 录制、
#                         取流缓冲与统计；LogFBank 前端、滚动语谱图、通用文本/路径工具
#
#  依赖方向：
#    本库不依赖 audio_ai_process，也不依赖 ORT / FreeType；
#    第三方依赖统一从仓库根 3rd/ 取：HCNetSDK -> 3rd/hk_sdk。
#
#  构建（独立工程，不需要先构建另一个库）：
#    cmake -S hk_mic_client -B hk_mic_client/build && cmake --build hk_mic_client/build -j
#  安装：
#    cmake --install hk_mic_client/build   # 3rd/zl_3rd/hk_mic_client/{include,lib}
##################################################################################

cmake_minimum_required(VERSION 3.10)
project(hk_mic_client LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# 仓库根：本工程在 <根>/hk_mic_client，3rd/ 与 results/ 都相对它定位
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(THIRD_DIR "${REPO_ROOT}/3rd")

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

# HCNetSDK：头文件与三平台预编译库统一放在 3rd/hk_sdk
set(HIK_SDK_ROOT     "${THIRD_DIR}/hk_sdk")
set(HIK_INCLUDE_DIRS "${HIK_SDK_ROOT}/include")
set(HIK_LIB_ROOT     "${HIK_SDK_ROOT}/lib/${HIK_LIB_SUBDIR}")

if(NOT EXISTS "${HIK_SDK_ROOT}/include/HCNetSDK.h")
    message(FATAL_ERROR "HCNetSDK 头文件缺失：${HIK_SDK_ROOT}/include/HCNetSDK.h")
endif()
if(NOT EXISTS "${HIK_LIB_ROOT}/libhcnetsdk${HIK_LIB_SUFFIX}")
    message(FATAL_ERROR "HCNetSDK 库缺失：${HIK_LIB_ROOT}/libhcnetsdk${HIK_LIB_SUFFIX}")
endif()

# 音频链路只需要这三个库；其余 .so 是它们的运行期依赖，通过 RPATH 找到
set(HIK_LIBS
    ${HIK_LIB_ROOT}/libhcnetsdk${HIK_LIB_SUFFIX}
    ${HIK_LIB_ROOT}/libHCCore${HIK_LIB_SUFFIX}
    ${HIK_LIB_ROOT}/libPlayCtrl${HIK_LIB_SUFFIX}
)

# ---------------------------------------------------------------------
# OpenCV：语谱图画布与入口里的 imshow / waitKey
# ---------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

find_package(Threads REQUIRED)

# =====================================================================
#  一、库：采集 + 特征工程
# =====================================================================
add_library(hk_mic_client SHARED
    src/hk_sdk_audio_client.cpp
    src/audio_resampler.cpp
    src/audio_preprocess.cpp
    src/spectrogram_image.cpp
    src/text_utils.cpp
)

# 输出 build/lib/lib_hk_mic_client.so（target 名保持 hk_mic_client，链接方不用改）；
# 安装后靠 $ORIGIN 找同目录的 HCNetSDK 运行库
set_target_properties(hk_mic_client PROPERTIES
    OUTPUT_NAME "_hk_mic_client"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "$ORIGIN"
)

# 公共头文件平铺在 include/，使用方直接 #include "xxx.h"；
# HCNetSDK 头与 OpenCV 头随 PUBLIC 传给使用方（入口会直接 include）。
target_include_directories(hk_mic_client PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${HIK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# 语谱图接口返回 / 接收 cv::Mat，OpenCV 属于公共依赖
target_link_libraries(hk_mic_client PUBLIC ${OpenCV_LIBS})
# HCNetSDK 只在本库内部使用，不把厂商库暴露给使用方
target_link_libraries(hk_mic_client PRIVATE ${HIK_LIBS} Threads::Threads)

set_target_properties(hk_mic_client PROPERTIES
    BUILD_RPATH "${HIK_LIB_ROOT}"
)

# =====================================================================
#  二、可执行入口（本库自给自足，不链接 audio_ai_process）
# =====================================================================
add_executable(save_audio_wav.out save_audio_wav.cpp)
add_executable(test_spectrogram_cv_mic.out test_spectrogram_cv_mic.cpp)
add_executable(test_spectrogram_cv_wav.out test_spectrogram_cv_wav.cpp)
add_executable(fbank_dump.out tools/fbank_dump.cpp)

foreach(TARGET
        save_audio_wav.out
        test_spectrogram_cv_mic.out
        test_spectrogram_cv_wav.out
        fbank_dump.out)
    # 入口用 HK_PROJECT_DIR 拼 results/ 等绝对路径；指向仓库根
    target_compile_definitions(${TARGET} PRIVATE
        HK_PROJECT_DIR="${REPO_ROOT}"
    )

    target_link_libraries(${TARGET} PRIVATE hk_mic_client ${OpenCV_LIBS} Threads::Threads)

    # 入口运行时要找到 3rd/hk_sdk 下的厂商库
    set_target_properties(${TARGET} PROPERTIES
        BUILD_RPATH "${HIK_LIB_ROOT}"
    )
endforeach()

# =====================================================================
#  三、安装：3rd/zl_3rd/hk_mic_client/{include,lib}
# =====================================================================
set(INSTALL_ROOT "${THIRD_DIR}/zl_3rd/hk_mic_client")

install(TARGETS hk_mic_client
    LIBRARY DESTINATION "${INSTALL_ROOT}/lib"
    ARCHIVE DESTINATION "${INSTALL_ROOT}/lib"
    RUNTIME DESTINATION "${INSTALL_ROOT}/bin"
)

# 公共头文件平铺；wav_reader.h 是入口共用的 RIFF 解析，也随库发布
install(FILES
    include/hk_sdk_audio_client.h
    include/audio_resampler.h
    include/audio_preprocess.h
    include/spectrogram_image.h
    include/text_utils.h
    include/wav_reader.h
    DESTINATION "${INSTALL_ROOT}/include"
)

# 公共头文件 hk_sdk_audio_client.h 直接 include "HCNetSDK.h"，
# 因此 SDK 头文件与运行库必须随库安装，否则使用方无法编译 / 运行。
install(DIRECTORY "${HIK_INCLUDE_DIRS}/"
    DESTINATION "${INSTALL_ROOT}/include"
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
)
install(DIRECTORY "${HIK_LIB_ROOT}/"
    DESTINATION "${INSTALL_ROOT}/lib"
    FILES_MATCHING
    PATTERN "*.so*"
    PATTERN "*.dll"
)
