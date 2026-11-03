#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预下载 ONNX Runtime 1.18.0 的构建依赖，并把 cmake/deps.txt 改写为本地路径。

为什么需要：CMake 在 --update 阶段用 FetchContent 从 GitHub 拉 abseil、onnx、
protobuf、cutlass 等约 40 个包，国内网络会中途被掐断（curl status 56）。
deps.txt 的 URL 列本身支持本地路径，所以先把包下载到本地并校验 SHA1，
再把 URL 改成绝对路径，配置阶段就完全不需要网络。

用法：
    python3 -I fetch_ort_deps.py

脚本只依赖标准库和 curl；路径、镜像都写死在下面常量里。
下载可中断重跑：已存在且 SHA1 正确的文件会跳过，只补缺失的。
任何一条失败时退出码为 1，且不会改写 deps.txt。
"""

import hashlib
import os
import subprocess
import sys
import time

# —— 一、路径与镜像 ——
# 源码目录：开发机先下好、Jetson 上重跑校验，两处路径写在一起按存在性选择，
# 这样同一份脚本两边都能直接用，不用改常量。
_SRC_CANDIDATES = (
    "/home/liyun/code/AI/3rd/onnxruntime-1.18.0",   # WSL 开发机（先下载）
    "/mnt/data/code/3rd/onnxruntime-1.18.0",        # Jetson 构建机
)
SRC_DIR = next((p for p in _SRC_CANDIDATES if os.path.isdir(p)), "")
# 依赖包缓存目录：放在本脚本所在的 onnxruntime/install/onnxruntime-1.18.0/ 下，
# 随 install 目录一起拷到 Jetson，删源码时不连带丢包。
# 用脚本自身位置拼路径，开发机和 Jetson 的项目根不同也能各自落到正确位置。
# 注意：把本脚本拷到别的目录运行（例如直接放进 ORT 源码目录），路径会跟着变，
# 包就会落在那个目录的 install/... 下；要固定位置请把下面这一行改成显式绝对路径。
DEPS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "install", "onnxruntime-1.18.0", "ort_deps",
)
# 镜像前缀：只作用于 github.com 的 URL；eigen 在 gitlab.com，走直连
# 备选镜像：https://gh-proxy.com/  https://ghproxy.net/
MIRROR = "https://ghfast.top/"

# —— 二、SHA1 覆盖（上游 deps.txt 记录失效时用） ——
# eigen：GitLab 重新生成过归档，同一个 commit（e7248b26...）的 zip 字节变了，
# ORT 1.18.0 deps.txt 记录的 be8be39f 已对不上，CMake 自己下也会校验失败。
# 实测 GitLab 当前归档（32b145f5）与 GitHub 镜像同 commit 归档解压后文件树
# 零差异，只是打包元数据不同，故信任实测值；改写 deps.txt 时会写入新值。
SHA1_OVERRIDES = {
    "eigen": "32b145f525a8308d7ab1c09388b2e288312d8eba",
}


def sha1_of(path):
    """按 1 MB 分块计算文件 SHA1。

    分块而不是一次 read()，避免 cutlass 这类上百 MB 的包占满内存。
    """
    digest = hashlib.sha1()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url, dest, want_sha, attempts=5):
    """下载 url 到 dest 并校验 SHA1，全部尝试失败时返回 False。

    用外层重试循环而不是 curl 的 --retry-all-errors：Ubuntu 20.04 自带的
    curl 7.68 没有这个参数，而 GitHub 掐断报的是 exit 56，--retry 不覆盖。
    每次尝试重下完整文件；先写 .part，校验通过才改名，失败不留半个正式文件。
    """
    tmp = dest + ".part"
    for i in range(1, attempts + 1):
        if os.path.exists(tmp):
            os.remove(tmp)
        subprocess.run(
            ["curl", "-fL", "--connect-timeout", "30",
             "--speed-limit", "1024", "--speed-time", "30", "-o", tmp, url],
            check=False,
        )
        if os.path.exists(tmp) and sha1_of(tmp) == want_sha:
            os.replace(tmp, dest)
            return True
        print(f"  第 {i}/{attempts} 次失败，重试中…", flush=True)
        time.sleep(2)
    if os.path.exists(tmp):
        os.remove(tmp)
    return False


def main():
    if not SRC_DIR:
        print("[错误] 以下候选目录都不存在，先确认 3.3.3 已解压：")
        for p in _SRC_CANDIDATES:
            print(f"  {p}")
        return 1
    deps_file = os.path.join(SRC_DIR, "cmake", "deps.txt")
    if not os.path.isfile(deps_file):
        print(f"[错误] 找不到 {deps_file}，先确认 3.3.3 已解压到 {SRC_DIR}")
        return 1
    os.makedirs(DEPS_DIR, exist_ok=True)

    # 备份只做一次，避免重复执行时把已改写的版本覆盖成"备份"
    backup = deps_file + ".orig"
    if not os.path.exists(backup):
        with open(deps_file, "r", encoding="utf-8") as src, open(backup, "w", encoding="utf-8") as dst:
            dst.write(src.read())

    new_lines = []
    failed = []
    downloaded = 0

    for raw in open(deps_file, "r", encoding="utf-8"):
        line = raw.rstrip("\n")
        # 注释行、空行原样保留
        if not line or line.startswith("#"):
            new_lines.append(raw)
            continue

        parts = line.split(";")
        if len(parts) != 3:
            print(f"[跳过] 无法解析的行：{line}")
            new_lines.append(raw)
            continue
        name, url, want_sha = parts
        # deps.txt 记录失效的条目改用实测值；改写时一并写入，CMake 才能通过
        if name in SHA1_OVERRIDES and want_sha != SHA1_OVERRIDES[name]:
            print(f"[覆盖] {name} 的 SHA1：{want_sha} -> {SHA1_OVERRIDES[name]}")
            want_sha = SHA1_OVERRIDES[name]

        # 已经改写成绝对路径（重复执行）时直接沿用。
        # 换机器后路径可能失效（开发机改写过的 deps.txt 拷到 Jetson），
        # 文件不存在要报失败而不是让 sha1_of 抛异常。
        if url.startswith("/"):
            dest = url
            if not os.path.isfile(dest):
                print(f"[失败] {name}：本地文件不存在 {dest}（可先还原 deps.txt.orig 再重跑）")
                failed.append(name)
            elif sha1_of(dest) != want_sha:
                print(f"[失败] {name}：本地文件 SHA1 不匹配 {dest}")
                failed.append(name)
            new_lines.append(raw)
            continue

        # 文件名带依赖名做前缀，避免不同仓库的 v3.0.1.zip 之类重名互相覆盖
        dest = os.path.join(DEPS_DIR, f"{name}-{os.path.basename(url)}")
        if os.path.exists(dest) and sha1_of(dest) == want_sha:
            print(f"[已存在] {name}")
        else:
            dl_url = MIRROR + url if url.startswith("https://github.com/") else url
            print(f"[下载] {name} <- {dl_url}", flush=True)
            downloaded += 1
            if not fetch(dl_url, dest, want_sha):
                print(f"[失败] {name}：下载或 SHA1 校验不通过，可换 MIRROR 后重跑")
                failed.append(name)
                continue

        new_lines.append(f"{name};{dest};{want_sha}\n")

    if failed:
        print(f"\n共 {len(failed)} 项失败：{', '.join(failed)}")
        print("deps.txt 未改写，换镜像后重跑本脚本即可（已完成的会跳过）")
        return 1

    with open(deps_file, "w", encoding="utf-8") as fp:
        fp.writelines(new_lines)
    print(f"\n完成：本次下载 {downloaded} 个，deps.txt 已全部指向 {DEPS_DIR}")
    print(f"原始文件保留在 {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
