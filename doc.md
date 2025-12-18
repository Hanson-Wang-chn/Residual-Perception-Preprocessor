# Build Project "Residual Perception Preprocessor"

## 项目背景

### Summary

我是一名具身智能研究生，正在研究manipulation方向。现在，我希望发表RSS顶会论文，距离现在还有一个月。我收集了一些真机数据（bimanual arms with grippers，有一个main camera和两个wrist camera，无Depth信息），任务名为“Sweep to Shapes”——通过毛刷（两只手都是一样的毛刷）sweep一大堆最小号的乐高积木，使之变为字母形状。具体方法是先把积木大致平铺成一个矩形，然后通过sweep“挖去”空心部分，然后再填上部分轮廓（如果需要）（全都是双臂协同的）。通过这个方法，我收集了"Z""E""N"三个字母的成功数据，每个字母15个trajectory，每个trajectory大约3分钟。我希望用该数据（如果数据量不足可以适当继续收集）微调VLA（比如π0.5），让机器人学会这一系列动作（主要是sweep），并结合不同的人language instruction，在常见的字母、数字上泛化。

### Policy Structure & Primitives Format

```
\begin{enumerate}
    \item 整体policy结构：
        \begin{itemize}
            \item Overall Input: language instruction, binary goal image, observations 
            \item Overall Output: action chunks
            \item High Level Policy (HLP): Finetuned VLM
                \begin{itemize}
                    \item Input: language instruction, binary goal image, observations
                    \item Output: primitive
                \end{itemize}
            \item Low Level Policy (LLP): Finetuned VLA ($\pi_0$ or $\pi_{0.5}$)
                \begin{itemize}
                    \item Input: primitive, observations
                    \item Output: action chunks
                \end{itemize}
            \item (Optional) Failure Detection Module: VLM
                \begin{itemize}
                    \item Input: observations (initial \& final), primitive
                    \item Output: Refine or not
                \end{itemize}
        \end{itemize}
    
    \item Primitives Format:
       \begin{itemize}
            \item 扫开特定位置的积木（坐标归一化到[0, 1000]中的整数）：
                \begin{itemize}
                    \item \lstinline{<Sweep> <Box> <x1, y1, x2, y2> <to> <Position> <x4, y4>}
                    \item \lstinline{<Sweep> <Triangle> <x1, y1, x2, y2, x3, y3> <to> <Position> <x4, y4>}
                \end{itemize}
            \item 清除特定范围内的杂乱积木：
                \begin{itemize}
                    \item \lstinline{<Clear> <Box> <x1, y1, x2, y2>}
                \end{itemize}
            \item 微调凌乱的积木：
                \begin{itemize}
                    \item \lstinline{<Refine> <Line> <x1, y1, x3, y3>}
                    \item \lstinline{<Refine> <Arc> <x1, y1, x2, y2, x3, y3>}
                \end{itemize}
        \end{itemize}
    
\end{enumerate}
```

### Current Concern: High Level Policy (HLP)

现在，我希望实现HLP的预处理部分。即，把输入的 image_goal 恰到好处地叠加到大致呈现矩形的积木图片上，然后分析出哪些地方的积木需要扫开。我希望尽可能降低后续Sweep的难度（比如减少需要Sweep的块数，让字母的一边靠着积木的一边，等等），且通过代码端到端地实现这一小步。

具体实现的Pipeline如下：

- 该模块的输入是来自摄像机主视角的图像 image_main_camera。
- 首先，使用最新的 Segment Anything Model (SAM) 3 处理 image_main_camera，把积木（scattered red Lego blocks）和桌面、机械臂等其他元素分开，得到 mask_lego。通过数字图像处理方法，对这个 mask_lego 做平滑操作（可以指定超参数）。
- 将 image_goal（参考`data/E.png`，指定文件路径）转换成二值图像 image_goal_binary。然后找到一个变换矩阵 $T$ (平移 $x, y$, 旋转 $\theta$, 缩放 $s$)，将 image_goal_binary 变换成 mask_goal_image，加到到 mask_lego 上。变换矩阵 $T$ 通过优化得到，参考下面的形式化部分。注意这一步需要可以指定不同的优化器（并分别实现），其中必须包括离散网格查找。
- 根据`render`参数，渲染并输出叠加后的图片。当`render == None`时，输出原始图片 image_main_camera；当`render == "goal"`时，输出image_main_camera加上颜色为{color1}的半透明 mask_goal_image；当`render == "residual"`时，输出image_main_camera加上颜色为{color2}的半透明 (mask_lego - mask_goal_image)；当`render == "goal+residual"`时，输出image_main_camera加上颜色为{color1}的半透明 mask_goal_image 和颜色为{color2}的半透明 (mask_lego - mask_goal_image)。{color1}和{color2}均可以通过RGB值指定，默认{color1}为亮绿色，{color2}为亮紫色。

变换矩阵 $T$ 通过优化得到，形式化如下：

$$
\min_{T} J(T) = \lambda_1 \cdot C_{fill} + \lambda_2 \cdot C_{remove} + \lambda_3 \cdot C_{edge} + \lambda_4 \cdot C_{sweep}
$$
其中 $\lambda_i$ 为各优化项的权重系数。
### 符号定义：
*   $M_{pile}$: 积木堆的二值掩码（1为积木，0为空）。
*   $M_{goal}(T)$: 经变换 $T$ 后的目标形状二值掩码。
*   $D_{pile}(p)$: 像素点 $p$ 在积木堆中的距离场值（到最近边界的距离）。
*   $E_{pile}$: 积木堆的边缘像素集合。
*   $E_{goal}(T)$: 变换后目标形状的边缘像素集合。
### 优化项详解：
**1. 填充惩罚 (Fill Cost) - $C_{fill}$**
惩罚需要“无中生有”填补积木的区域。因为抓取零散积木进行填充比直接扫除要慢，应尽量减少。
$$
C_{fill} = \sum_{p} \mathbb{I}(p \in M_{goal}(T) \land p \notin M_{pile})
$$
*解释：计算目标掩码中有而积木掩码中没有的像素面积。*
**2. 移除惩罚 (Removal Cost) - $C_{remove}$**
惩罚需要扫除的积木总量。操作越少，效率越高。
$$
C_{remove} = \sum_{p} \mathbb{I}(p \notin M_{goal}(T) \land p \in M_{pile})
$$
*解释：计算积木掩码中有而目标掩码中没有的像素面积（即废料）。*
**3. 边缘重合奖励 (Edge Alignment Cost) - $C_{edge}$**
鼓励目标形状的轮廓利用现有的积木堆轮廓，这样可以减少修整边缘的精细操作（Refine primitives）。
$$
C_{edge} = - \sum_{p \in E_{goal}(T)} \exp\left(-\frac{\min_{q \in E_{pile}} \|p - q\|^2}{2\sigma^2}\right)
$$
*解释：利用 Chamfer Distance 的变体。对于目标边缘的每个点，寻找最近的积木边缘点。距离越近，奖励越大（Cost 越负）。*
**4. 扫除难度惩罚 (Sweepability Cost) - $C_{sweep}$**
这是降低 Manipulation 难度的关键。我们希望需要移除的部分位于积木堆的“浅层”或边缘，而不是被包围在中心（那样需要先把外层扫开才能处理内层）。
$$
C_{sweep} = \sum_{p} \left[ \mathbb{I}(p \notin M_{goal}(T) \land p \in M_{pile}) \cdot D_{pile}(p)^\alpha \right]
$$
*解释：对于每一个需要移除的像素，乘以其距离场值。如果一个需要移除的点位于积木堆深处（$D_{pile}$ 大），则惩罚呈指数级增长（$\alpha > 1$）。这会迫使算法倾向于让目标形状覆盖积木堆的中心，而将需要切除的部分留在边缘。

## 你需要完成的任务

用Python代码实现“Current Concern: High Level Policy (HLP)”中提到的HLP预处理部分的Pipeline。必须确保代码可以正确运行。

## 具体要求

- SAM3的代码在`sam3/`中，在当前环境下已经可以正常运行（参考`sam3/inference_test.py`）。注意模型权重文件在`/home/whs/manipulation/Residual-Perception-Preprocessor/sam3/sam3.pt`，不要从 Huggingface 下载。
- 计算 (mask_lego - mask_goal_image) 时，如果mask_goal_image中包含mask_lego中没有的像素，忽略即可。
- 使用 `opencv-python` 库。
