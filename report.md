# Gradient Boosted Decision Trees for High Dimensional Sparse Output

# 应对高维稀疏输出的梯度提升决策树

### 背景介绍

GBDT是常见的基于决策树的算法，用于分类与回归任务。在这篇文章中，作者主要探讨关于高维稀疏环境下的GBDT。

需要理解GBDT-SPARSE需要先理解GBDT，需要理解GBDT就需要先理解CART决策树。决策树认为，物以类聚、人以群分，在特征空间里相近的样本，那就是一类。如果为每个“类”分配的空间范围比较小，那么，同一个类内的样本差异会非常小，以至于看起来一样。换句话说，如果我们可以将特征空间切分为较小的碎块，然后为每一个碎块内的样本配置一个统一的因变量取值，就有机会做出误差较小的预测。这也是分类和聚类算法的基本假设。

决策树有很多种，CART是其中最常见的之一。它是一个二叉树，也就是每个节点需要更具一定条件将属于它的样本分为两个子集并给两个子集预测值。

举一个例子，比如下图我们要切分父节点。它包含$D$这一样本集。

![cart.drawio](https://gitee.com/zhuhaojia2001/pic_go_pics/raw/master/img/cart.drawio.png)

（树的某个节点）

可以很自然的想到切分条件是一个数值对：
$$
t=[feature, threhold]
$$
在选定的$feature$上，大于阈值$threhold$的样本给$D_r$，小于阈值的给$D_l$。

分配好了后就需要分别给左右分别一个预测值，在这个两个预测值下我们希望损失是最小的。损失函数用的是square error。那就可以写出目标函数：
$$
\min _{j, x}\left[\min _{c_2} \sum_{x_{1} \in D_l}\left(y_{i}-c_{1}\right)^{2}+\min _{c_{2}} \sum_{x_{1} \in D_r}\left(y_{i}-c_{2}\right)^{2}\right]
$$
也就是遍历所有可能的$t$，算出对应的最小预测值，然后计算损失。随后取最小的损失来进行切分。

对于GBDT来说，整个模型可以视作一个由众多弱分类器组成的强分类器。如下图所示：

![image-20211215193951570](https://gitee.com/zhuhaojia2001/pic_go_pics/raw/master/img/image-20211215193951570.png)

（GBDT）

把问题定义为以下形式：
$$
X=\left\{x_{i}\right\}_{i=1}^{N} x_{i}\quad \in R^{D}\\
Y=\left\{y_{i}\right\}_{i=1}^{N}\quad y_{i} \in R^{L}\quad y_{i} \in\{0,1\}^L
$$


每个树是一个弱分类器，将所有的弱分类器的预测结果相加可以得到强分类器，可以写成以下这个式子：
$$
F(x)=\sum_{m=1}^{T} f_{m}(x)
$$
最小化损失可以表示为：
$$
F^{*}=\underset{F}{\operatorname{argmin}} \sum_{i=1}^{N} \mathcal{L}\left(y_{i}, F\left(x_{i}\right)\right)+R(F)
$$
其中正则项可展开：
$$
R(F)=\lambda \sum_{m=1}^{T} \sum_{j=1}^{M_{m}}\left\|w_{j}^{m}\right\|^{2}
$$
这个正则项是为了防止过拟合并让模型快速收敛的。

在训练阶段每个树的训练作为一个阶段，第$m$阶段，固定前$m-1$个树，拟合第$m$棵树。虽然大多情况用CART作为弱分类器，但GBDT为了适应多种损失函数，把这时的目标函数泰勒展开为以下形式：
$$
\begin{aligned}
&\mathcal{L}\left(y_{i}, F_{m-1}\left(x_{i}\right)+f_{m}\left(x_{i}\right)\right) \approx \\
&\mathcal{L}\left(y_{i}, F_{m-1}\left(x_{i}\right)\right)+\left\langle g_{i}, f_{m}\left(x_{i}\right)\right\rangle+\frac{1}{2}\left\|f_{m}\left(x_{i}\right)\right\|^{2}
\end{aligned}
$$

为了让这个式子小，需要第$m$棵树去拟合负梯度。也就是：
$$
g_{i}=-\left.\frac{\partial \mathcal{L}\left(y_{i}, F\left(x_{i}\right)\right)}{\partial F\left(x_{i}\right)}\right|_{F\left(x_{i}\right)=F_{m-1}\left(x_{i}\right)}\\
\arg \min _{f_{m}} \sum_{i=1}^{N} \frac{1}{2}\left(f_{m}\left(x_{i}\right)-g_{i}\right)^{2}
$$
第$m$阶段需要解的问题就是：
$$
\begin{aligned}
&\min _{f_{m}, w_{j}^{m}} \sum_{i=1}^{N}\left\|g_{i}-f_{m}\left(x_{i}\right)\right\|_{2}^{2}+\lambda \sum_{j=1}^{M_{m}}\left\|w_{j}^{m}\right\|_{2}^{2} \\
\end{aligned}
$$
这里论文里有点问题$g$没写负号，可以按99年的论文来。在解决了拟合什么的问题后，就需要进入到具体怎么切分，其实就是加上正则项，这里就简略写了。目标方程为：
$$
\operatorname{obj}(t)=\frac{1}{N} \sum_{i \in V_{e}}\left\|g_{i}-h_{e, i}\right\|^{2}+\lambda\left(\left\|h_{r}\right\|^{2}+\left\|h_{l}\right\|^{2}\right)
$$
$h_r,h_s$为左右子节点的预测值，每次切分需要选出能让这个目标函数最小的$t$。

往往在现实运用中，样本的数量往往在百万级别，并且样本特征和标签也同样百万级别。那就会产生以下问题：

1. $g$每棵树都要算，假设有$N$个样本每个样本L维度，那更新$NL$个参数太多了
2. 每片叶子在训练的时候都要存相对的样本，空间复杂度为$O(TML)$数量太多了
3. 时间复杂度太高，每个样本的时间复杂度是$O(Tl+TL)^2$
4. $x$太稀疏了，导致树很不平衡而且有一堆叶节点

本篇的主要贡献就是提出加入$L_0$正则项，同时用更先进的GBDT-SPARSE算法来改善在稀疏情况下的表现。



### 实验平台

- Windows 10 
- `python=3.8, numpy=1.18.5, pandas`



### 算法流程

- **集成方法**

像传统的GBDT集成方法一样，首先设置弱学习器的个数，在每次的迭代过程中，使用本轮的弱学习器去拟合目标函数在当前模型上的负梯度，即训练一个决策树，它的输入为本轮的训练样本，目标为之前模型结果总和与真实目标之间的“残差”，通过不断地增加弱学习器来减少这种“残差”进而最终拟合真实目标。

与传统GBDT每个样本对应一个单独的目标不同，我们每个样本都对应一个L维的向量作为目标，如果像传统GBDT直接扩展到多分类那样为目标的每一维建立一颗树，那么随着L的增长，所有树的归纳时间以及所占空间都是无法忍受的。所以我们直接以高维向量为输出，并且保证它们是稀疏的，从而使每次计算梯度的时候都能得到稀疏的向量。

- **树的构建**

像基本的CART树一样，每个树从根结点开始，每次选择一个数据属性和一个属性值对当前样本集合进行分割，将样本划为两部分分别进入左右两个子节点，子节点继续进行划分，直到达到最大树深度或者当前节点样本数少于设定值。

像集成方法中所说的那样，我们对训练样本只归纳一棵树，并希望得到多维的输出。在树归纳完成后，每个叶子节点会产生一个预测向量，它由之前介绍的公式，根据此叶节点中的所有样本的本轮目标(即负梯度信息)计算得出，它便是一个L维向量。集成模型最后的输出是稀疏的，因而我们也要保证每个预测向量也都是稀疏的。



GBDT-SPARSE首先在每阶段的预测上加入$L_0$约束。具体就是对于$m$步加入$L_0$约束项，如下：
$$
\begin{aligned}
&\min _{f_{m}, w_{j}^{m}} \sum_{i=1}^{N}\left\|g_{i}-f_{m}\left(x_{i}\right)\right\|_{2}^{2}+\lambda \sum_{j=1}^{M_{m}}\left\|w_{j}^{m}\right\|_{2}^{2} \\
&\text { s.t. }\left\|w_{j}^{m}\right\|_{0} \leq k, \quad \forall j 
\end{aligned}
$$
这样就限制了预测向量中只有$k$项是非零，其他全部强制置零。这对高度稀疏的数据来说可以避免不必要的计算。也就是说需要选出$k$个最稠密的标签保留。衡量稠密的方法就是设定$p_{q}^{l}=\sum_{i \in V_{l}}\left(g_{i}\right)_{q}$，然后对所有$p$排序：
$$
\left|p_{\pi(1)}^{l}\right| \geq\left|p_{\pi(2)}^{l}\right| \geq \ldots \geq\left|p_{\pi\left(\left|V_{l}\right|\right)}^{l}\right|
$$
选前$k$个，其他全部置零：
$$
\left(h_{l}\right)_{q}^{*}= \begin{cases}p_{q}^{l} /\left(\left|V_{l}\right|+\lambda\right) & \text { if } \pi(q) \leq k \\ 0 & \text { otherwise }\end{cases}
$$

具体算法如下：

![algorithm](https://gitee.com/zhuhaojia2001/pic_go_pics/raw/master/img/algorithm.png)

（GBDT-SPARSE算法）

像之前公式介绍的那样，我们使用 $p$ 来衡量所有样本的梯度信息在每维上的稀疏程度，通过得到的目标函数的闭式解来计算每个划分方案的优劣，这个过程中使用了 $L_0$ 正则化来保证稀疏，同时计算过程中得到 $p$ 的值可以帮助叶节点计算预测向量。

在算法实现过程中，与算法中所写稍有不同，为了防止过拟合，我们可以不遍历每个样本 $i$ 的属性值进行划分测试，而是设定一个测试间隔 $S$ ，从而对 $x_{\sigma(i)}, x_{\sigma(i+S)}, ...$ 进行测试。在训练样本数 $N$ 比较大的时候我们常取 $S = \frac{N}{20}$。这可以在降低过拟合风险的同时降低计算时间。

-------

**经过这样拓展后得到的GBDT，可以直接实现高维输出，并且在保证稀疏度约束的情况下减小了树的存储空间以及归纳时间，同时由稀疏性限制，也可以较快的进行预测。可以作为工具较好地应用到多标签分类的任务中。除此之外，作为传统GBDT的多输出扩展，理论上将其应用到多目标的拟合任务（L较小）或是多分类任务中（稀疏度k=1）都是可以的。**

-----

作者在论文中提到，决策树方法在样本数据特征稀疏时，会导致一系列问题，比如树十分不平衡、树非常深等问题，为了解决这种问题，作者使用额外的降维技术对数据进行预处理，得到稠密的特征形式，再使用GBDT-Sparse进行拟合。具体比较了随机投影、PCA、LEML等方法，由于LEML属于监督方法，可以融合高维标签信息构造数据投影矩阵，因而具有更好的效果。由于这些方法都是额外工作，并且只用于特征稀疏数据的预处理操作，就不再过多解释。

### 实验结果

![he_learn_rate_score](https://gitee.com/zhuhaojia2001/pic_go_pics/raw/master/img/he_learn_rate_score.png)

（he数据集learnrate和各种score的关系）

![re_learn_rate_score](https://gitee.com/zhuhaojia2001/pic_go_pics/raw/master/img/re_learn_rate_score.png)

（re数据集learnrate和各种score的关系）

![he_max_depth_score](https://gitee.com/zhuhaojia2001/pic_go_pics/raw/master/img/he_max_depth_score.png)

（he数据集max_depth和各种score的关系）

![re_max_depth_score](https://gitee.com/zhuhaojia2001/pic_go_pics/raw/master/img/re_max_depth_score.png)

（re数据集max_depth和各种score的关系）

HE和RE都是max_depth=5，learn_rate=1，yeast是max_depth=3，learn_rate=1.0，Delicious上使用的max_depth=10，learn_rate=10.8

|                | HE     | RE     | yeast  |
| -------------- | ------ | ------ | ------ |
| f1_score_macro | 0.4622 | 0.7001 | 0.6097 |
| f1_score_micro | 0.7424 | 0.8495 | 0.7563 |
| accuracy       | 0.6266 | 0.6883 | 0.3478 |
| hamming_loss   | 0.135  | 0.0958 | 0.2919 |

|           | P@1    | P@3    |
| --------- | ------ | ------ |
| Delicious | 0.7048 | 0.6986 |



### 实验分析/总结

通过实验所得结果与组内其他多标签分类算法在相同数据集上所得结果进行对比，可以看出本算法可以得到相近或更好的指标，因此认为我们所实现的GBDT_Sparse算法可以完成基本的多标签分类任务，并且使用论文中提到的针对极端多标签分类的Delicious数据集进行了实验，得到的指标接近论文中的值，也验证了我们实现的算法在高维稀疏输出场景下的可用性。针对训练和预测的加速优化问题，由于作者并未放出源码，并且提到使用C++进行实现，而我们只是基于python和numpy的简单实现，对可能涉及到多线程优化、GPU加速等开发知识不太了解，所以可能不能达到论文所说的速度方面的更大优势。

本文是对传统GBDT在高维稀疏输出场景时的推广，主要的应用场景就是极端多标签学习等，并且针对高维稀疏输出做了一系列优化，使得其可以得到一定准确性的同时保持其高效性，并且作为首个将GBDT推广到高维稀疏输出场景的工作，也为后来的一些研究提供了不少的启发。它比较适用于大规模的高维稀疏数据，并且更加针对极端情况。就多标签学习任务而言，这个模型对损失函数还是有限制，并且需要像GBDT一样针对不同数据集调整很多参数。在之后的发展中，我们认为还可以将其推广到更加普遍的高维输出情况，并且像更新的树方法中使用2阶梯度信息等对任意形式的损失函数进行优化，同时之后也发现有利用梯度直方图等对计算进行加速的方法。并且针对多标签学习，可以考虑更多的结合标签之间的关系来获得更好的预测。

### 参考

```latex
@inproceedings{si2017gradient,
  title={Gradient boosted decision trees for high dimensional sparse output},
  author={Si, Si and Zhang, Huan and Keerthi, S Sathiya and Mahajan, Dhruv and Dhillon, Inderjit S and Hsieh, Cho-Jui},
  booktitle={International conference on machine learning},
  pages={3182--3190},
  year={2017},
  organization={PMLR}
}
```

```
@article{friedman2002stochastic,
  title={Stochastic gradient boosting},
  author={Friedman, Jerome H},
  journal={Computational statistics \& data analysis},
  volume={38},
  number={4},
  pages={367--378},
  year={2002},
  publisher={Elsevier}
}
```
