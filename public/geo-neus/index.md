# Geo-Neus: Geometry-Consistent Neural Implicit Surfaces Learning for Multi-view Reconstruction

## 公式推导
用有符号距离函数表示空间点p到物体表面的距离，sdf(p)等于0
$$\partial \Omega=\{\boldsymbol{p} \mid s d f(\boldsymbol{p})=0\}$$

使用两个神经网络估计SDF和颜色场
$$s \hat{d} f(\boldsymbol{p})=F_{\Theta}(\boldsymbol{p})$$
$$\hat{c}(\boldsymbol{o}, \boldsymbol{v}, t)=G_{\Phi}(\boldsymbol{o}, \boldsymbol{v}, t)$$

表示颜色场：
$$
C=\hat{C}=\sum_{i=1}^n w\left(t_i\right) \hat{c}\left(t_i\right)
$$
由于需要专注于表面的重建，所以将公式改写为
$$
\begin{aligned}
C & =\sum_{i=1}^{j-1} w\left(t_i\right) \hat{c}\left(t_i\right)+w\left(t_j\right) \hat{c}\left(\hat{t^*}\right)+w\left(t_j\right)\left(\hat{c}\left(t_j\right)-\hat{c}\left(\hat{t^*}\right)\right)+\sum_{i=j+1}^n w\left(t_i\right) \hat{c}\left(t_i\right) \\
& =w\left(t_j\right) \hat{c}\left(\hat{t^*}\right)+\varepsilon_{\text {sample }}+\sum_{\substack{i=1 \\
i \neq j}}^n w\left(t_i\right) \hat{c}\left(t_i\right) \\
& =w\left(t_j\right) \hat{c}\left(\hat{t^*}\right)+\varepsilon_{\text {sample }}+\varepsilon_{\text {weight }},
\end{aligned}
$$
![公式推导](posts/paper-geo-neus/tuidao.JPEG "公式推导")

最后得到偏差的表达式：
$$
\Delta c=\hat{c}\left(\hat{t}^*\right)-c\left(t^*\right)=\frac{\left(1-w\left(t_j\right)\right) c\left(t^*\right)-\varepsilon_{\text {sample }}-\varepsilon_{\text {weight }}}{w\left(t_j\right)}
$$
$$
\delta c=\frac{\Delta c}{c\left(t^*\right)}=\frac{1}{w\left(t_j\right)}-1-\frac{\varepsilon_{\text {sample }}+\varepsilon_{\text {weight }}}{w\left(t_j\right) c\left(t^*\right)} .
$$

网络的架构图，结合了三种loss，五个通道
![架构图](posts/paper-geo-neus/structure.png "架构图")

闭塞处理，这里没有看懂在干什么
Occlusion handling. Because we focus on opaque objects, some parts of objects are invisible from view of a certain camera position. Therefore, there are only some of the sparse points visible for each view. For an image $I_i$ with camera position $\boldsymbol{o}_i$, the visible points $\boldsymbol{P}_i$ are consistent with feature points $\boldsymbol{X}_i$ of $I_i$
$$
\boldsymbol{X}_i=\boldsymbol{K}_i\left[\boldsymbol{R}_i \mid \boldsymbol{t}_i\right] \boldsymbol{P}_i
$$
where $\boldsymbol{K}_i$ is the internal calibration matrix, $\boldsymbol{R}_i$ is the rotation matrix and $\boldsymbol{t}_i$ is the translation vector for image $I_i$. The coordinates of $\boldsymbol{X}_i$ and $\boldsymbol{P}_i$ are all homogeneous coordinates. The scale index before $\boldsymbol{X}_i$ is omitted for simplicity. According to feature points of each image, we get visible points for each view and use them to supervise the SDF network while rendering image from the corresponding view.

SDF loss
View-aware SDF loss. While rendering image $I_i$ from view $V_i$, we use the SDF network to estimate SDF values for the visible points $\boldsymbol{P}_i$ of $V_i$. Based on the approximation that the SDF values of sparse points are zeroes, we propose the view-aware SDF loss:
$$
\mathcal{L}_{S D F}=\sum_{\boldsymbol{p}_j \in \boldsymbol{P}_i} \frac{1}{N_i}\left|s \hat{d} f\left(\boldsymbol{p}_j\right)-s d f\left(\boldsymbol{p}_j\right)\right|=\sum_{\boldsymbol{p}_j \in \boldsymbol{P}_i} \frac{1}{N_i}\left|s \hat{d} f\left(\boldsymbol{p}_j\right)\right|,
$$
where $N_i$ is the number of points in $\boldsymbol{P}_i$ and $|\cdot|$ denotes the $L_1$ distance. It is worth noting that the loss we use to supervise the SDF network varies according to the view being rendered. In this way, the introduced SDF loss is consistent with the process of color rendering.
值得注意的是，我们用来监督SDF网络的损失根据所呈现的视图而有所不同。这样，引入的SDF损失与显色渲染的过程是一致的。
这个loss应该向0收敛吗？是的
这里提到了他的模型使用了几何先验，是使用了什么作为先验？

表达隐式平面，通过正负距离确定表面点。使用线性插值的方法求交点集。
Occlusion-aware implicit surface capture. We use the implicit representation of the surface, and extract surface with the zero-level set of the implicit function. So the question is: Where is our implicit surface? According to Formula (3), the estimated surface is:
$$
\hat{\partial \Omega}=\{\boldsymbol{p} \mid s \hat{d} f(\boldsymbol{p})=0\} .
$$
We aim to optimize $\hat{\partial \Omega}$ with geometry-consistent constraints among different views. Because the number of points on the surface is infinite, we need to sample points from $\hat{\partial \Omega}$ in practice. To maintain consistency with the process of color rendering using view rays, we sample the surface points on these rays. As mentioned in 3.1, we sample $t$ discretely along the view ray and use the Riemann sum to obtain the rendered colors. Based on the sampled points, we use linear interpolation to get the surface points.

With sampled point $t$ on the ray, the corresponding 3D point is $\boldsymbol{p}=\boldsymbol{o}+t \boldsymbol{v}$, and the predicted SDF value is $s \hat{d} f(\boldsymbol{p})$. For simplicity, we further represent $s \hat{d} f(\boldsymbol{p})$ as $s \hat{d} f(t)$, which is the function of $t$. We find the sample point $t_i$, the sign of whose SDF value is different from the next sample point $t_{i+1}$. The sample points set $T$ formed by $t_i$ is:
$$
T=\left\{t_i \mid s \hat{d} f\left(t_i\right) \cdot s \hat{d} f\left(t_{i+1}\right)<0\right\} .
$$
In this situation, the line $t_i t_{i+1}$ intersects with the surface $\hat{\partial \Omega}$. The intersection points set $\hat{T}^*$ is:
$$
\hat{T}^*=\left\{t \mid t=\frac{s \hat{d} f\left(t_i\right) t_{i+1}-s \hat{d} f\left(t_{i+1}\right) t_i}{s \hat{d} f\left(t_i\right)-s \hat{d} f\left(t_{i+1}\right)}, t_i \in T\right\}
$$

多视角的几何约束：

使用MVS中的光学一致性来监督隐式表面的生成
MVS：MVS（Multi-View Stereo）的目的是在已知相机位姿的前提下估计稠密的三维结构，如果相机位姿未知则会先使用SfM得到相机的位姿。评估的稠密结构真值一般是基于Lidar（ETH3D[4]、Tanks and Temples[5]）或者深度相机（DTU[6]）获得整个场景的点云，而对应的相机位姿有的在采集的时候就通过机械臂直接获取[6]，有的则基于采集的深度进行估计[4][5]。
有了场景点云之后，一般评估指标为精度和完整度，同时还有二者的平衡指标F1分数。
1) 精度（Accuracy）：对于每个估计出来的3D点寻找在一定阈值内的真值3D点，最终可以匹配上的比例即为精度。需要注意的是，由于点云真值本身不完整，需要先估计出真值空间中不可观测的部分，估计精度时忽略掉。
2) 完整度（Completeness）：将每个真值的3D点寻找在一定阈值内最近的估计出来的3D点，最终可以匹配上的比例即为完整度。
3) F1分数（F1 Score）：精度和完整度是一对trade-off的指标，因为可以让点布满整个空间来让完整度达到100%，也可以只保留非常少的绝对精确的点来得到很高的精度指标。因此最终评估的指标需要对二者进行融合。假设精度为p，完整度为r，则F1分数为它们的调和平均数，即2pr / (p + r)。

Multi-view photometric consistency constraints. We capture our estimated implicit surface, of which the geometric structures are supposed to be consistent among different views. Based on this intuition, we use the photometric consistency constraints in multi-view stereo (MVS) [8, 9, 34] to supervise our extracted implicit surface.

For a small area $s$ on the surface, the projection of $s$ on the image is a small pixel patch $q$. The patches corresponding to $s$ are supposed to be geometry-consistent among different views, except for occlusion occasions. Similar to patch warping in traditional MVS methods, we use the central point and its normal to represent $s$. For convenience, we represent the plane equation of $s$ in the camera coordinate of the reference image $I_r$ :
$$
\boldsymbol{n}^T \boldsymbol{p}+d=0,
$$
where $\boldsymbol{p}$ is the intersection point computed through Formula (19) and $\boldsymbol{n}^T$ is the normal computed with automatic differentiation of SDF network at $\boldsymbol{p}$. Then the image point $\boldsymbol{x}$ in the pixel patch $q_i$ of reference image $I_r$ is related to the corresponding point $\boldsymbol{x}^{\prime}$ in the pixel patch $q_{i s}$ of the source image $I_s$ via the plane-induced homography $\boldsymbol{H}[11]$ :
$$
\boldsymbol{x}=\boldsymbol{H} \boldsymbol{x}^{\prime}, \boldsymbol{H}=\boldsymbol{K}_s\left(\boldsymbol{R}_s \boldsymbol{R}_r^T-\frac{\boldsymbol{R}_s\left(\boldsymbol{R}_s^T \boldsymbol{t}_s-\boldsymbol{R}_r^T \boldsymbol{t}_r\right) \boldsymbol{n}^T}{d}\right) \boldsymbol{K}_r^{-1}
$$
where $\boldsymbol{K}$ donates the internal calibration matrix, $\boldsymbol{R}$ donates the rotation matrix and $\boldsymbol{t}$ donates the translation vector. The index indicates which image the donation belongs to. To concentrate on the geometric information, we convert the color images $\left\{I_i\right\}$ into gray images $\left\{I_i^{\prime}\right\}$, and supervise our implicit surface with the photometric consistency among patches in $\left\{I_i^{\prime}\right\}$.

然后得出几何一致损失
Photometric consistency loss. To measure the photometric consistency, we use the normalization cross correlation (NCC) of patches in the reference gray image $\left\{I_r^{\prime}\right\}$ and the source gray image $\left\{I_s^{\prime}\right\}$ :
$$
N C C\left(I_r^{\prime}\left(q_i\right), I_s^{\prime}\left(q_{i s}\right)\right)=\frac{\operatorname{Cov}\left(I_r^{\prime}\left(q_i\right), I_s^{\prime}\left(q_{i s}\right)\right)}{\sqrt{\operatorname{Var}\left(I_r^{\prime}\left(q_i\right)\right) \operatorname{Var}\left(I_s^{\prime}\left(q_{i s}\right)\right)}},
$$
where Cov denotes covariance and $\operatorname{Var}$ donates variance. While rendering colors for an image, we use the patches which take the pixels being rendered as center and the patch size is $11 \times 11$. We take the rendered image as the reference image and compute NCC scores between its sampled patches and their corresponding patches on all source images. To handle occlusions, we find the best four of the computed NCC scores for each sampled patch following [9], and use them to compute the photometric consistency loss for the corresponding view:
$$
\mathcal{L}_{\text {photo }}=\frac{\sum_{i=1}^N \sum_{s=1}^4 1-N C C\left(I_r^{\prime}\left(q_i\right), I_s^{\prime}\left(q_{i s}\right)\right)}{4 N},
$$
where $N$ is the number of sampled pixels on the rendered image. With the photometric consistency loss, the geometric consistency of the implicit surface among multiple views is guaranteed.

总体的损失函数：
During rendering colors from a specific view, our total loss is:
$$
\mathcal{L}=\mathcal{L}_{\text {color }}+\alpha \mathcal{L}_{\text {reg }}+\beta \mathcal{L}_{S D F}+\gamma \mathcal{L}_{\text {photo }} .
$$
$\mathcal{L}_{\text {color }}$ is the difference between the ground truth colors and the rendered colors:
$$
\mathcal{L}_{\text {color }}=\frac{1}{N} \sum_{i=1}^N\left|C_i-\hat{C}_i\right| .
$$
And $\mathcal{L}_{r e g}$ is an eikonal term [10] to regularize the gradients of SDF network:
$$
\mathcal{L}_{\text {reg }}=\frac{1}{N} \sum_{i=1}^N\left(\left|\nabla s \hat{d} f\left(\boldsymbol{p}_i\right)\right|-1\right)^2 .
$$
In our experiments, we choose $\alpha, \beta$ and $\gamma$ as $0.3,1.0$ and $0.5$ respectively.

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
