---
weight: 4
title: "Geo-Neus: Geometry-Consistent Neural Implicit Surfaces Learning for Multi-view Reconstruction"
date: 2023-01-04T21:57:40+08:00
lastmod: 2023-01-09T17:11:40+08:00
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
## 公式推导
用有符号距离函数表示空间点p到物体表面的距离，sdf(p)等于0
![1](1.PNG)

使用两个神经网络估计SDF和颜色场
![2](2.PNG)

表示颜色场：
![3](3.PNG)

由于需要专注于表面的重建，所以将公式改写为
![公式推导](tuidao.JPEG "公式推导")

最后得到偏差的表达式：
![4](4.png)

网络的架构图，结合了三种loss，五个通道
![架构图](structure.png "架构图")

闭塞处理，这里没有看懂在干什么
![5](5.png)

SDF loss
View-aware SDF loss. While rendering image $I_i$ from view $V_i$, we use the SDF network to estimate SDF values for the visible points $\boldsymbol{P}_i$ of $V_i$. Based on the approximation that the SDF values of sparse points are zeroes, we propose the view-aware SDF loss:
![6](6.png)
值得注意的是，我们用来监督SDF网络的损失根据所呈现的视图而有所不同。这样，引入的SDF损失与显色渲染的过程是一致的。
这个loss应该向0收敛吗？是的
这里提到了他的模型使用了几何先验，是使用了什么作为先验？

表达隐式平面，通过正负距离确定表面点。使用线性插值的方法求交点集。
![7](7.png)
![8](8.png)
![9](9.png)
多视角的几何约束：

使用MVS中的光学一致性来监督隐式表面的生成
MVS：MVS（Multi-View Stereo）的目的是在已知相机位姿的前提下估计稠密的三维结构，如果相机位姿未知则会先使用SfM得到相机的位姿。评估的稠密结构真值一般是基于Lidar（ETH3D[4]、Tanks and Temples[5]）或者深度相机（DTU[6]）获得整个场景的点云，而对应的相机位姿有的在采集的时候就通过机械臂直接获取[6]，有的则基于采集的深度进行估计[4][5]。
有了场景点云之后，一般评估指标为精度和完整度，同时还有二者的平衡指标F1分数。
1) 精度（Accuracy）：对于每个估计出来的3D点寻找在一定阈值内的真值3D点，最终可以匹配上的比例即为精度。需要注意的是，由于点云真值本身不完整，需要先估计出真值空间中不可观测的部分，估计精度时忽略掉。
2) 完整度（Completeness）：将每个真值的3D点寻找在一定阈值内最近的估计出来的3D点，最终可以匹配上的比例即为完整度。
3) F1分数（F1 Score）：精度和完整度是一对trade-off的指标，因为可以让点布满整个空间来让完整度达到100%，也可以只保留非常少的绝对精确的点来得到很高的精度指标。因此最终评估的指标需要对二者进行融合。假设精度为p，完整度为r，则F1分数为它们的调和平均数，即2pr / (p + r)。

![10](10.png)
然后得出几何一致损失

![11](11.png)

总体的损失函数：
![12](12.png)

## 代码复现
>GitHub网址：[点击跳转](https://github.com/GhiXu/Geo-Neus)
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

5. 对loss function具体内容的再阐述  
   整个网络的输入是稀疏点和多视角照片（可以参考网络架构图）  
    ```python
    def forward(self, inputs):
        """
        :type input_rgb: object
        """
        inputs = inputs * self.scale

        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
    ```
  * sdf网络中的forward函数其实返回的就是点的sdf值，对应文章中所说的MLP
  * 监督过程发生在渲染的同时，渲染的同时我们会返回一个点的sdf值，那么我们已有的三维稀疏点也可以进行渲染，利用sdf网络来返回sdf值，因为我们假设三维稀疏点在表面，因此我们试着用网络来返回，如果网络准确，它应该返回一个接近于0的值，同时为了弥补三维稀疏点的缺陷，也进行一个视角射线的抽样，利用多视角的照片，进行操作抽取点，这些是中间的过程，就是所谓的隐式曲面的抽取。同样为了监督隐式曲面的抽取，光度一致性损失被提出，这些都是中途的过程。综合来说，输入只有三位稀疏点和各个角度的照片。
   ```python
      # Loss
      color_error = (color_fine - true_rgb) * mask
      color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
      psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())
      
      eikonal_loss = gradient_error
      
      mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
      
      sdf_loss = F.l1_loss(pts2sdf, torch.zeros_like(pts2sdf), reduction='sum') / pts2sdf.shape[0]
      
      ncc_loss = 0.5 * (ncc_cost.sum(dim=0) / (inside_sphere.sum(dim=0) + 1e-8)).squeeze(-1)
      
      loss = color_fine_loss + eikonal_loss * self.igr_weight + mask_loss * self.mask_weight + sdf_loss + ncc_loss
   ```
* 以上是loss的具体计算过程

6. renderer.py 代码解读
```python
   def get_psnr(img1, img2, normalize_rgb=False):
    if normalize_rgb: # [-1,1] --> [0,1]
        img1 = (img1 + 1.) / 2.
        img2 = (img2 + 1. ) / 2.

    mse = torch.mean((img1 - img2) ** 2)
    psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).cuda())

    return psnr
```
- 这个函数计算两张图像img1和img2的峰值信噪比（PSNR）。如果normalize_rgb标志设置为True，函数将通过将范围从[-1,1]转换为[0,1]来标准化图像。然后，函数通过计算两张图像的像素值之差的平方平均值来计算两幅图像之间的均方误差（MSE）。然后使用MSE和最大可能像素值（对于8位图像为255）的log base 10计算PSNR。返回最终的PSNR值。


```python
  def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u
  ```

- 这个函数通过在由边界框坐标和分辨率定义的网格上的点上调用查询函数来提取字段。该函数首先将网格分割成大小为N = 64的块，然后使用torch.linspace()函数生成一组线性间隔值，用于x，y和z维。然后，函数使用嵌套for循环遍历网格的块，对于每个块，它使用该块中的点调用查询函数，并将结果存储在u数组中。
- 这个函数使用numpy库创建数组，Pytorch创建点网格并将它们分块处理。

```python
def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles
```

- 这个函数通过使用Marching Cubes算法来从3D场中提取几何体。它接受四个参数：bound_min和bound_max是3D空间的下限和上限，resolution是每个维度上的点数，threshold是Marching Cubes算法用来确定场中哪些点被认为是几何体的一部分的值，query_func是返回场在一组点上的值的函数。
- 该函数首先调用extract_fields()函数获取一个3D场，然后将场和阈值值传递给mcubes库中的marching_cubes()函数，该函数返回提取的几何体的顶点和三角形。然后，缩放并偏移顶点以匹配由bound_min和bound_max指定的边界框。
- 该函数使用Pytorch处理数据，使用mcubes库提取几何体，并使用Pytorch将张量转换为numpy数组。