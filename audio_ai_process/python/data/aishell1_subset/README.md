---
license: Apache License 2.0
tags:
- asr
- Aishell-1
audio:
  auto-speech-recognition:
    languages:
      - zh-cn
    sampling_rates:
      - 16000
---
# 仅用于单元测试

## 数据集描述
用于“Aishell-1学术数据集的中文语音识别模型”的Aishell-1测试集


### 数据集简介
用于“Aishell-1学术数据集的中文语音识别模型”的Aishell-1测试集，该测试集并非完整的Aishell-1数据集。

希尔贝壳中文普通话开源语音数据库AISHELL-ASR0009-OS1录音时长178小时，是希尔贝壳中文普通话语音数据库AISHELL-ASR0009的一部分。AISHELL-ASR0009录音文本涉及智能家居、无人驾驶、工业生产等11个领域。录制过程在安静室内环境中， 同时使用3种不同设备： 高保真麦克风（44.1kHz，16-bit）；Android系统手机（16kHz，16-bit）；iOS系统手机（16kHz，16-bit）。高保真麦克风录制的音频降采样为16kHz，用于制作AISHELL-ASR0009-OS1。400名来自中国不同口音区域的发言人参与录制。经过专业语音校对人员转写标注，并通过严格质量检验，此数据库文本正确率在95%以上。分为训练集、开发集、测试集。（支持学术研究，未经允许禁止商用。）

### 数据集支持的任务
支持“Aishell-1学术数据集的中文语音识别模型”模型的测试任务

## 数据集的格式和结构

### 数据格式
16K采样率，单通道音频。


### 数据集加载方式
无

### 数据分片
无需分片，全部为test音频



## 数据集生成的相关信息

### 原始数据
无

### 数据集标注
无

#### 标注过程
无

#### 标注者
无



## 数据集版权信息
版权归aishell所有，支持学术研究，未经允许禁止商用。

## 引用方式
```BibTeX
@inproceedings{aishell_2017,
  title={AIShell-1: An Open-Source Mandarin Speech Corpus and A Speech Recognition Baseline},
  author={Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, Hao Zheng},
  booktitle={Oriental COCOSDA 2017},
  pages={Submitted},
  year={2017}
}
```

## 其他相关信息
详见https://www.openslr.org/33/


### Clone with HTTP
```bash
git clone https://www.modelscope.cn/datasets/modelscope/aishell1_subset.git
```
