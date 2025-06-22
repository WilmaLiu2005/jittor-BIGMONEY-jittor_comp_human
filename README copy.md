# jittor_comp_human

计图比赛骨骼绑定赛题

# 前置知识

在计算机图形学中，骨骼动画是重要的研究对象。本赛题将在经典的骨骼表示和linear blend shape算法框架下预测一个网格的骨骼节点（joint）。

一个网格 $\mathcal{M}$（mesh）能被顶点 $\mathcal{V}$ 和三角面片 $\mathcal{F}$ 刻画，其中 $\mathcal{V}\in\mathbb{R}^{n\times 3}$ 为包含 $n$ 个三维顶点的矩阵， $\mathcal{F}\in\mathbb{N}^{f\times 3}$ 为描述了每个三角面片包含的顶点下标的矩阵。

那么，为了让网格动起来，需要额外添加什么东西？首先运动的肯定是网格上的顶点，为方便起见不考虑顶点和顶点之间的影响。也就是说需要找到一个关于顶点 $x$ 的函数 $f$ ， $f_{\mathcal{M}}(x,M)$ 能够表示网格 $\mathcal{M}$ 上的顶点 $x$ ，在条件 $M$ 下时最后变换到的位置。如果只考虑一些简单的形变，例如平移、旋转，那么这些信息可以由一个 $4\times 4$ 的仿射矩阵$M$表示。

其次，当谈到形变时，不得不考虑它究竟是在哪一个坐标系中进行的，也就是说需要描述顶点究竟是关于那个局部坐标系，进行了如何的变换。这里一个简单的想法是引入局部坐标系在全局坐标系中的坐标 $x,y,z$ 。但是描述“顶点绕局部坐标系旋转90°”时，需要知道顶点是朝着哪个方向旋转的，就需要记录局部坐标系的 $x,y,z$ 轴的方向，也就是说局部坐标系也可以被表示成一个 $4\times 4$ 的仿射矩阵 $M_{c}$ 。

同时，大家希望顶点能够收到多个局部坐标系的控制，那么不同的局部坐标系对同一个顶点的影响是如何的？一种朴素的想法是把他们的效果线性叠加，同时系数归一化。

最后，受到人体关节的启发，局部坐标系可以影响到另一部分局部坐标系，但他们的影响不应该成环。也就是说局部坐标系的影响关系应该形成一棵“树”。对树中的每个节点赋值基础的变换后，子树中的每个节点都可以根据其祖先和自己的基础变换得到最后的变换值。

那么，形式化地，一个网格可以规定他的骨架 $\mathcal{J}$ ，每个顶点关于骨架的权重 $\mathcal{S}$ ，每个joint的父亲节点 $\mathcal{P}\in\{\mathbb{N}\cup\{-1\}\}$ ，其中 $\mathcal{J}\in\mathbb{R}^{J\times 3}$ 为包含了 $J$ 个joint的坐标的矩阵， $\mathcal{S}\in\mathbb{R}^{n\times J}$ 描述了每个节点关于这 $J$ 个joint的权重的矩阵。第 $i$ 个joint有初始的状态 $M_{local,i}$ 、终止的状态 $M_{pose,i}$ ，那么最后第 $j$ 个顶点 $v_j(x, y, z, 1)$ 的位置即为：
```math
f_{\mathcal{M}}(v_j, M)=\sum_{i=1}^{J} \mathcal{S}_{j, i} M_{pose,i} * M_{local_i}^{-1}v_j
```

其中乘法表示矩阵乘法，且 $\sum_{i=1}^{j}v_{j,i}=1$（归一化）。 $\mathcal{P}$ 表示了父亲节点的下标（-1表示根节点，没有父亲），并且有：

```math
M_{pose,i}=M_{pose,\mathcal{P}_{i}} * \left( M_{local,\mathcal{P}_{i}}^{-1} *M_{local,i} \right)  * M_{basis,i}
```

其中 $M_{basis_i}$ 表示第 $i$ 个joint的基础变换矩阵，默认 $M_{pose,-1},M_{local,-1}$ 为单位矩阵。并且 $M_{pose}$ 应该递归计算。

**如果选手不明白上面在说什么，也不必担心具体实现。baseline中已经给出了上述计算的全部实现。**

本赛题中，选手需要根据给定数据集，预测出网格的骨架 $\mathcal{J}$ 的**坐标**（不包括坐标轴的方向）和权重 $\mathcal{S}$ 。为方便起见，骨架中joint的个数总是一个固定的值，骨架的连接关系 $\mathcal{P}$ 也是固定的。

## 运行环境

首先确保电脑上安装了conda进行坏境管理。在terminal中依次运行以下代码：

```
conda create -n jittor_comp_human python=3.9
conda activate jittor_comp_human
conda install -c conda-forge gcc=10 gxx=10 # 确保gcc、g++版本不高于10
pip install -r requirements.txt
```

## 数据下载

[点击下载](https://cloud.tsinghua.edu.cn/f/676c582527f34793bbac/?dl=1)

下载后将其解压缩到当前根目录。

## baseline介绍

baseline分为预测joint、预测skin两个阶段。所有输入的数据均被归一化到了`[-1, 1]^3`中（包括joint，选手需多加注意），并且在mesh表面通过随机顶点+混合均匀采样的方式得到`(batch, n, 3)`的点云。

在第一阶段中，模型使用PointTransformer [1](#ref1) 将输入的形状为`(batch, n, 3)`的三维点云变成了形状为`(batch, feat_dim)`的隐向量，之后通过一个`MLP`得到形状为`(batch, 66)`的输出（66=3*22，22为固定的joint个数），通过reshape操作便得到了最终预测的joint。

在第二阶段中，模型先使用PointTransformer得到形状为`(batch, feat_dim)`的形状隐向量，再将其分别用于两个`MLP`，得到了每个点和joint的`query`向量，接着通过类似于交叉注意力 [2](#ref2) 的方式得到每个点关于每个joint的得分，最后通过`softmax`得到了每个点关于每个joint的权重。

可以看到以上网络的实现仍然是比较朴素的，我们希望选手能在此基础上探索更具有泛化性能的网络来获得更好的效果。

## 运行baseline

运行baseline训练代码：

```
bash launch/train_skeleton.sh
bash launch/train_skin.sh
```

其中，`train_skeleton`会运行骨骼预测训练，在一张4090上需要2GB显存，大致运行2小时。train_skin会运行蒙皮预测训练，在一张4090上需要略大于2GB显存，大致运行2小时。模型会保存在`output`文件夹下。

同时，一些临时在`validate`集上的预测结果会输出在`tmp`文件夹中，选手可以查看其中的内容来判断训练是否大致正确。最终会占用13GB的空间，因此选手应该预留足够多的空间或选择减少输出中间的可视化结果。

## 预测并提交结果

运行预测代码：

```
bash launch/predict_skeleton.sh
bash launch/predict_skin.sh
```

预测的结果会输出在`predict`中。

如果需要可视化预测结果，可以运行以下代码（需要下面的debug环境）：

```
# 只渲染图片结果
bash launch/render_predict_results.sh

# 不渲染图片结果，但是导出fbx文件
bash launch/render_predict_results.sh --render False --export_fbx True
```

渲染的结果将会输出到`tmp_predict`中。

选手最终需要提交以下结果：

```
predict
└   vroid
│   ├   2011
│   │   ├   predict_skeleton.npy
│   │   ├   predict_skin.npy
│   │   └   transformed_vertices.npy
│   ├   2012
│   │   ├   predict_skeleton.npy
│   │   ├   predict_skin.npy
│   │   └   transformed_vertices.npy
│   ...
└mixamo
    ├   3189
    │   ├   ...
    ...
```
其中，`vroid`文件夹中必须包含`data/test_list.txt`中的所有`cls`为`vroid`的待预测文件，`mixamo`文件夹中必须包含`data/test_list.txt`中的所有`cls`为`mixamo`的待预测文件。

`predict_skeleton.npy`是预测的骨骼数据，包含形状为`(J, 3)`的数据，每行的含义参考`dataset/format.py`。

`predict_skin.npy`是预测的蒙皮数据，包含形状为`(N, J)`的数据，其中第`i`行的`J`个数字对应了原本的`mesh`的第`i`个节点，分别关于骨骼的`J`个蒙皮权值。

`transformed_vertices.npy`是选手预测的骨骼对应的顶点坐标，包含形状为`(N, 3)`的数据。选手*要确保*预测的骨架能够对应这个顶点数据（也就是说，对于原本顶点坐标的`transform`操作，和对于预测骨骼的`transform`操作要完全一致）。在评测时会将原本`mesh`的顶点和选手给出的`transformed_vertices`归一化到`[-1, 1]^3`中进行评测。评测指标包括`joint to joint loss`、`vertex loss`、`skin l1 loss`、`vertex normalization loss`。

## 可视化

本赛题的可视化比较困难，因此需要使用特殊的环境来进行debug和可视化。

首先安装环境：

```
conda create -n jittor_comp_human_debug python=3.11
conda activate jittor_comp_human_debug
pip install -r requirements_debug.txt
```

在`dataset/exporter.py`中的`Exporter`里提供了一系列可视化的操作。选手可以参考`debug_example.py`中的文件。

除了查看直接的渲染结果（以`_render`开头的api），选手可以使用以下方式查看`obj`等结果：

1. 在vscode中安装相应的3d浏览插件，例如`vscode-3d-preview`。

2. 使用一些建模软件，例如`blender`。

## 参考文献

<a name="ref1"></a> [PCT: Point Cloud Transformer](https://arxiv.org/abs/2012.09688)

<a name="ref2"></a> [Attention Is All You Need](https://arxiv.org/abs/1706.03762)