##################################################################################
#  audio_ai_processor —— 语音处理 ROS 节点
#
#  职责：
#    海康设备实时取流 → LogFBank 前端 → 双模式 KWS/ASR 解码，
#    把唤醒 / 结束语事件与识别文本发布成 std_msgs/String。
#
#  依赖：
#    - ROS：roscpp、std_msgs
#    - 项目内库（audio_ai_process 仓库的安装产物，默认 3rd/zl_3rd）：
#        libhk_mic_client.so       采集 + 前端 + 语谱图
#        libaudio_ai_process.so    模型解码 + 双模式 Session
#    - ONNX Runtime：按架构从 3rd/onnxruntime/install 选择（与库编译时一致）
#    - 系统库：OpenCV、Threads
#
#  构建前先准备好上面两个库的安装产物：
#    cd <audio_ai_process 仓库>
#    cmake -S hk_mic_client -B hk_mic_client/build && cmake --build hk_mic_client/build -j
#    cmake --install hk_mic_client/build
#    cmake -S audio_ai_process -B audio_ai_process/build && cmake --build audio_ai_process/build -j
#    cmake --install audio_ai_process/build
#
#  本包位于 <仓库根>/catkin_ws/src/audio_ai_processor，CMake 默认向上三级
#  定位仓库根（仓库整体搬目录不用改）；特殊布局下用
#    catkin_make -DAUDIO_AI_ROOT=/path/to/audio_ai_process
#  覆盖（RPATH 与 ORT 路径都跟着它走）。
##################################################################################

cmake_minimum_required(VERSION 3.0.2)
project(audio_ai_processor)

# 与工作区其它节点保持一致：C++17 + Release
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

# 语谱图可视化与画布类型（show_window=true 时才用到，但链接期固定带上）
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------
# audio_ai_process 仓库根：库安装产物与权重都在这里
#   本包位于 <仓库根>/catkin_ws/src/audio_ai_processor，上三级就是仓库根，
#   整个仓库搬目录时不用改；特殊布局下用 catkin_make -DAUDIO_AI_ROOT=... 覆盖
# ---------------------------------------------------------------------
if(NOT DEFINED AUDIO_AI_ROOT)
  get_filename_component(AUDIO_AI_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
endif()
set(ZL3RD_DIR "${AUDIO_AI_ROOT}/3rd/zl_3rd")

set(HK_INCLUDE_DIR  "${ZL3RD_DIR}/hk_mic_client/include")
set(HK_LIBRARY      "${ZL3RD_DIR}/hk_mic_client/lib/libhk_mic_client.so")
set(ASR_INCLUDE_DIR "${ZL3RD_DIR}/audio_ai_process/include")
set(ASR_LIBRARY     "${ZL3RD_DIR}/audio_ai_process/lib/libaudio_ai_process.so")

if(NOT EXISTS "${HK_LIBRARY}")
  message(FATAL_ERROR
    "hk_mic_client 未安装：${HK_LIBRARY}\n"
    "  先在该仓库里执行：cmake -S hk_mic_client -B hk_mic_client/build && "
    "cmake --build hk_mic_client/build -j && cmake --install hk_mic_client/build\n"
    "  或用 -DAUDIO_AI_ROOT=<仓库路径> 指定正确的仓库根")
endif()
if(NOT EXISTS "${ASR_LIBRARY}")
  message(FATAL_ERROR
    "audio_ai_process 未安装：${ASR_LIBRARY}\n"
    "  先在该仓库里执行：cmake -S audio_ai_process -B audio_ai_process/build && "
    "cmake --build audio_ai_process/build -j && cmake --install audio_ai_process/build\n"
    "  或用 -DAUDIO_AI_ROOT=<仓库路径> 指定正确的仓库根")
endif()
message(STATUS "audio_ai_process 仓库: ${AUDIO_AI_ROOT}")

# ---------------------------------------------------------------------
# ONNX Runtime：与 audio_ai_process 库的编译口径一致（按架构选版本）
#   x86_64  -> onnxruntime-1.23.2
#   aarch64 -> onnxruntime-1.18.0
# 节点本身也直接调用 ORT（建 Ort::Session），所以头与库都要显式引入。
# ---------------------------------------------------------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(ORT_VERSION_DIR "onnxruntime-1.18.0")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
  set(ORT_VERSION_DIR "onnxruntime-1.23.2")
else()
  message(FATAL_ERROR "ONNX Runtime: unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(ORT_ROOT        "${AUDIO_AI_ROOT}/3rd/onnxruntime/install/${ORT_VERSION_DIR}")
set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
set(ORT_LIB_DIR     "${ORT_ROOT}/lib")
set(ORT_LIBRARY     "${ORT_LIB_DIR}/libonnxruntime.so")

if(NOT EXISTS "${ORT_LIBRARY}")
  message(FATAL_ERROR
    "ONNX Runtime 未安装：${ORT_LIBRARY}\n"
    "  x86_64  : 把预编译包复制到 3rd/onnxruntime/install/onnxruntime-1.23.2/\n"
    "  aarch64 : 按 3rd/onnxruntime/README.md 第 3 节在 Jetson 上编译并安装")
endif()

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${HK_INCLUDE_DIR}
  ${ASR_INCLUDE_DIR}
  ${ORT_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

# 节点本体：main 只做参数与发布编排，流水线在 audio_ai_processor_node
add_executable(${PROJECT_NAME}_node
  src/main.cpp
  src/audio_ai_processor_node.cpp
)

target_link_libraries(${PROJECT_NAME}_node PRIVATE
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${ASR_LIBRARY}
  ${HK_LIBRARY}
  ${ORT_LIBRARY}
  Threads::Threads
)

# 运行期动态库：
#   - 两个项目库 + 其依赖的 ORT 不在系统库路径里，必须写进 RPATH；
#   - HCNetSDK 的运行库与 libhk_mic_client.so 同目录（库自身 RPATH=$ORIGIN）。
set_target_properties(${PROJECT_NAME}_node PROPERTIES
  BUILD_RPATH
    "${ZL3RD_DIR}/hk_mic_client/lib;${ZL3RD_DIR}/audio_ai_process/lib;${ORT_LIB_DIR}"
  INSTALL_RPATH
    "${ZL3RD_DIR}/hk_mic_client/lib;${ZL3RD_DIR}/audio_ai_process/lib;${ORT_LIB_DIR}"
)

install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
