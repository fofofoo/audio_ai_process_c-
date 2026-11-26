##################################################################################
#  audio_ai_process —— 流式 ASR / KWS 模型处理库
#
#  职责：
#    lib_audio_ai_process.so  三模型 ORT 推理、贪心 / 束搜索解码、关键词前缀树、
#                            双模式 session 编排、中文文本渲染
#
#  依赖方向：
#    本库依赖 hk_mic_client（特征类型与公共头，链接其安装产物），
#    依赖 ONNX Runtime（3rd/onnxruntime）、OpenCV、FreeType（系统库）。
#    构建前需要先构建并安装 hk_mic_client：
#      cmake -S hk_mic_client -B hk_mic_client/build && \
#      cmake --build hk_mic_client/build -j && cmake --install hk_mic_client/build
#
#  构建：
#    cmake -S audio_ai_process -B audio_ai_process/build && \
#    cmake --build audio_ai_process/build -j
#  安装：
#    cmake --install audio_ai_process/build  # 3rd/zl_3rd/audio_ai_process/{include,lib}
##################################################################################

cmake_minimum_required(VERSION 3.10)
project(audio_ai_process LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# 仓库根：本工程在 <根>/audio_ai_process，3rd/、weight/、results/ 都相对它定位
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(THIRD_DIR "${REPO_ROOT}/3rd")
set(ZL3RD_DIR "${THIRD_DIR}/zl_3rd")

# ---------------------------------------------------------------------
# OpenCV：语谱图类型 / 入口 imshow（语谱图实现在 hk_mic_client，这里主要给入口）
# ---------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------
# ONNX Runtime：按架构选择项目内托管的版本
#   x86_64  -> 3rd/onnxruntime/install/onnxruntime-1.23.2
#   aarch64 -> 3rd/onnxruntime/install/onnxruntime-1.18.0
# ---------------------------------------------------------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ORT_VERSION_DIR "onnxruntime-1.18.0")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(ORT_VERSION_DIR "onnxruntime-1.23.2")
else()
    message(FATAL_ERROR "ONNX Runtime: unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(ORT_ROOT        "${THIRD_DIR}/onnxruntime/install/${ORT_VERSION_DIR}")
set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
set(ORT_LIB_DIR     "${ORT_ROOT}/lib")
set(ORT_LIBRARY     "${ORT_LIB_DIR}/libonnxruntime.so")

if(NOT EXISTS "${ORT_LIBRARY}")
    message(FATAL_ERROR
        "ONNX Runtime 未安装：${ORT_LIBRARY}\n"
        "  x86_64  : 把预编译包复制到 3rd/onnxruntime/install/onnxruntime-1.23.2/\n"
        "  aarch64 : 按 3rd/onnxruntime/README.md 第 3 节在 Jetson 上编译并安装")
endif()
message(STATUS "ONNX Runtime root : ${ORT_ROOT}")

# ---------------------------------------------------------------------
# hk_mic_client：使用其安装产物（头 + .so），不直接引用对方源码目录
# ---------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(HK_LIBRARY "${ZL3RD_DIR}/hk_mic_client/lib/_hk_mic_client.lib")
else()
    set(HK_LIBRARY "${ZL3RD_DIR}/hk_mic_client/lib/lib_hk_mic_client.so")
endif()
set(HK_INCLUDE_DIR "${ZL3RD_DIR}/hk_mic_client/include")

if(NOT EXISTS "${HK_LIBRARY}")
    message(FATAL_ERROR
        "hk_mic_client 未安装：${HK_LIBRARY}\n"
        "先执行：cmake -S hk_mic_client -B hk_mic_client/build && "
        "cmake --build hk_mic_client/build -j && cmake --install hk_mic_client/build")
endif()
message(STATUS "hk_mic_client     : ${ZL3RD_DIR}/hk_mic_client")

# 以 imported target 形式引入，后续链接/头文件路径都跟着它走
add_library(hk_mic_client SHARED IMPORTED)
set_target_properties(hk_mic_client PROPERTIES
    IMPORTED_LOCATION "${HK_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${HK_INCLUDE_DIR}"
)

# ---------------------------------------------------------------------
# FreeType：text_renderer.cpp 用系统字体渲染中文（cv::putText 只支持 ASCII）。
# 找不到时 text_renderer 编译成空实现，入口靠 ready() 回退 ASCII。
# ---------------------------------------------------------------------
find_package(Freetype QUIET)
if(FREETYPE_FOUND)
    message(STATUS "FreeType found: 中文识别结果用系统字体渲染")
else()
    message(STATUS "FreeType 未找到：识别结果窗口文字回退 ASCII")
endif()

# =====================================================================
#  一、库：ASR / KWS 模型处理
# =====================================================================
add_library(audio_ai_process SHARED
    src/asr_decode.cpp
    src/asr_session.cpp
    src/kws_asr_session.cpp
    src/kws_decode.cpp
    src/kws_context_graph.cpp
    src/text_renderer.cpp
)

# 输出 build/lib/lib_audio_ai_process.so（target 名保持 audio_ai_process，链接方不用改）
set_target_properties(audio_ai_process PROPERTIES
    OUTPUT_NAME "_audio_ai_process"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "$ORIGIN"
)

# 公共头文件平铺在 include/，使用方直接 #include "xxx.h"；
# 公共头 asr_decode.h / kws_decode.h 需要 audio_preprocess.h 的特征类型，
# 因此 hk_mic_client 的 include 路径与库都作为 PUBLIC 依赖传递下去。
target_include_directories(audio_ai_process PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${ORT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(audio_ai_process PUBLIC
    hk_mic_client
    ${ORT_LIBRARY}
    ${OpenCV_LIBS}
)
target_link_libraries(audio_ai_process PRIVATE Threads::Threads)

if(FREETYPE_FOUND)
    # ${TARGET}链接到 FreeType 的库（中文文本渲染）
    target_compile_definitions(audio_ai_process PRIVATE HK_HAVE_FREETYPE=1)
    target_include_directories(audio_ai_process PRIVATE ${FREETYPE_INCLUDE_DIRS})
    target_link_libraries(audio_ai_process PRIVATE ${FREETYPE_LIBRARIES})
endif()

# providers_cuda.so 由 ORT 运行时 dlopen，必须与 libonnxruntime.so 同目录，
# 因此 BUILD_RPATH 指向 ORT_LIB_DIR；hk_mic_client 同理。
set_target_properties(audio_ai_process PROPERTIES
    BUILD_RPATH "${ORT_LIB_DIR};${ZL3RD_DIR}/hk_mic_client/lib"
)

# =====================================================================
#  二、可执行入口
# =====================================================================
add_executable(test_wav_asr.out test_wav_asr.cpp)
add_executable(test_wav_kws.out test_wav_kws.cpp)
add_executable(test_mic_asr.out test_mic_asr.cpp)
add_executable(test_mic_kws_asr.out test_mic_kws_asr.cpp)
add_executable(load_onnx_model.out load_onnx_model.cpp)
add_executable(test_context_graph.out tools/test_context_graph.cpp)

set(EXEC_DEMOS
    test_wav_asr.out
    test_wav_kws.out
    test_mic_asr.out
    test_mic_kws_asr.out
    load_onnx_model.out
    test_context_graph.out
)

# 需要设备取流的入口：显式链接 hk_mic_client（库本身已 PUBLIC 传递，
# 这里写出来让每个入口的依赖一眼可见）
set(EXEC_LINK_HK
    test_mic_asr.out
    test_mic_kws_asr.out
)

foreach(TARGET ${EXEC_DEMOS})
    # 入口用 HK_PROJECT_DIR 拼 weight/、results/、keywords/ 的绝对路径；指向仓库根
    target_compile_definitions(${TARGET} PRIVATE
        HK_PROJECT_DIR="${REPO_ROOT}"
    )

    target_include_directories(${TARGET} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${TARGET} PRIVATE audio_ai_process ${OpenCV_LIBS} Threads::Threads)

    if(${TARGET} IN_LIST EXEC_LINK_HK)
        target_link_libraries(${TARGET} PRIVATE hk_mic_client)
    endif()

    # 入口运行时需要 ORT 与 hk_mic_client 的 .so
    set_target_properties(${TARGET} PROPERTIES
        BUILD_RPATH "${ORT_LIB_DIR};${ZL3RD_DIR}/hk_mic_client/lib"
        INSTALL_RPATH "${ORT_LIB_DIR};${ZL3RD_DIR}/hk_mic_client/lib"
    )
endforeach()

# =====================================================================
#  三、安装：3rd/zl_3rd/audio_ai_process/{include,lib}
# =====================================================================
set(INSTALL_ROOT "${ZL3RD_DIR}/audio_ai_process")

install(TARGETS audio_ai_process
    LIBRARY DESTINATION "${INSTALL_ROOT}/lib"
    ARCHIVE DESTINATION "${INSTALL_ROOT}/lib"
    RUNTIME DESTINATION "${INSTALL_ROOT}/bin"
)

install(FILES
    include/asr_decode.h
    include/asr_session.h
    include/block_worker.h
    include/kws_asr_session.h
    include/kws_context_graph.h
    include/kws_decode.h
    include/text_renderer.h
    DESTINATION "${INSTALL_ROOT}/include"
)
