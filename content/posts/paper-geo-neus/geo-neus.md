---
weight: 4
title: "Geo-Neus: Geometry-Consistent Neural Implicit Surfaces Learning for Multi-view Reconstruction"
date: 2023-1-04T21:57:40+08:00
lastmod: 2023-01-05T16:45:40+08:00
draft: false
author: "Hanqi Jiang"
authorLink: "https://github.com/hq0709"
description: "2022年最新发表的NeuS变种：Geo-NeuS"
images: []
# resources:
# - name: "featured-image"
#   src: "featured-image.png"

tags: ["3D reconstruction", "Deep learning"]
categories: ["papers"]

lightgallery: true
---

## 代码复现
1. conda activate报错

    ```shell
    CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
    ```
    解决方案：

    ```shell
    source activate # 执行后命令行出现base标识
    conda activate ...
    ```

2. conda activate pytorch 报错
    ```shell
    Could not find conda environment: pytorch
    You can list all discoverable environments with `conda info --envs`.
    ```
    
    ```shell
    conda install pytorch
    ```
    
    ```shell
    Executing transaction: failed
    ERROR conda.core.link:_execute(502): An error occurred while installing package 'conda-forge::certifi-2022.9.24-pyhd8ed1ab_0'.
    FileNotFoundError(2, "No such file or directory: '/home/hanqi/.conda/envs/geoneus/bin/python3.7'")
    
    Attempting to roll back.

    Rolling back transaction: done

    FileNotFoundError(2, "No such file or directory: '/home/hanqi/.conda/envs/geoneus/bin/python3.7'")
    ```
    默认的安装路径是错的，我把forge卸载了，之后可以正常安装
    ```shell
    anaconda search -t conda fvcore
    anaconda show ...
    ```

    不知道为什么，anaconda3文件夹设置的是read only，如果要进行更新操作必须切换root用户
    ```shell
    sudo s # 切换root用户
    exit # 退出回到原来的用户
    ```

3. 程序运行一段时间被kill了，定位发现是在这句话的时候
    ```python
    self.images_gray = torch.from_numpy(self.images_gray_np.astype(np.float32)).cuda()
    ```
    .cuda()是把数据储存到显卡里，可能是数据量太大了，我删除了一些照片，还剩十几张

4. 训练过程中报错缺失npy文件
    ```shell
    FileNotFoundError: [Errno 2] No such file or directory: 'pikachu_aruco/sfm_pts/points.npy'
    ```
    查找后发现是作者没有给出这个文件的计算方法，只是说要modify [这个文件](https://github.com/GhiXu/ACMP/blob/master/colmap2mvsnet_acm.py)
    运行指令需要两个，一个是数据文件夹，一个是保存的路径，直接运行报下面的错误
    ```shell
    FileNotFoundError: [Errno 2] No such file or directory: 'pikachu_aruco/sparse/cameras.txt'
    ```
    这个文件作者没有提供，也没有解释文件的内容是什么

5. fields.py
    分成四个class，SDF network, Rendering netwrk, SingleVariance network 和 NeRF