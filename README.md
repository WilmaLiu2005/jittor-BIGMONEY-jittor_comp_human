# 计图赛题 2：人体骨骼生成

## 项目概述

本项目利用一批 3D 人体网格模型以及对应的骨骼位置、蒙皮权重，依据 3D 人体骨骼格式介绍，对这些模型预测骨骼节点的空间位置以及蒙皮权重。即输入一个三维网格 M，需要预测出 J 个三维坐标表示骨骼节点的位置，以及 N×J 的矩阵表示原网格中 N 个点对于 J 个骨骼的蒙皮权重取值。

## 运行环境

首先确保电脑上安装了 conda 进行环境管理。在 terminal 中依次运行以下代码：

```
conda create -n jittor_comp_human python=3.9
conda activate jittor_comp_human
conda install -c conda-forge gcc=10 gxx=10 # 确保gcc、g++版本不高于10
pip install -r requirements.txt
```

## 数据下载

[点击下载](https://cloud.tsinghua.edu.cn/f/676c582527f34793bbac/?dl=1)
下载后将其解压缩到当前根目录，文件夹名为 data。

## 分支说明

不同分支是本组实现的不同的改进方法
LXY_1:skeleton 郑皓之改进 1 + skin 刘昕雨改进 1
LXY_2:skeleton 刘昕雨改进 2 + skin 刘昕雨改进 2
ZHZ_1:skeleton 郑皓之改进 1 + skin 郑皓之改进 1
ZHZ_2:skeleton 郑皓之改进 2 + skin 郑皓之改进 2

## 运行 baseline

运行训练代码：

```
bash launch/train_skeleton.sh
bash launch/train_skin.sh
```

每个任务在一张 4090 训练约需要 20 小时，在一张 A100 上训练约需要 1 天

## 预测并提交结果

运行预测代码：

```
bash launch/predict_skeleton.sh
bash launch/predict_skin.sh
```

预测的结果会输出在`predict`中。

## 项目经验

**架构的构建是困难的，需要足够的经验积累才能构建出比较好的模型**
**调参大约可以带来 3-5 分的涨幅**
