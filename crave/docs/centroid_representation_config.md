# CRAVE 簇中心代表图 · 标准配置(当前最优选择)

> 2026-06-18 定稿。经编码器扫 + 解码器倒 U 阶梯 + 数据/频率消融后,**确定簇中心可视化(解码质心)的标准 enc/dec 配置**。
> 详细实验留痕见 [center_representative_research.md](visualization/center_representative_research.md)(平均脸 deep research → patch 解码 → 对齐 → GAN → 规模消融 A–I → 解码器倒 U → ep2302 30Hz 演示)。

---

## ★ 标准配置(定为当前最优)

| 组件 | 选择 | 参数 | 理由 |
|---|---|---|---|
| **编码器** | **DINOv2-large** | 1024d / ~300M | 簇中心语义结构最丰富(112 vs base 89 / small 92) |
| **解码器** | **small**(空间卷积) | **0.92M** | 解码器倒 U 的**峰值**(tiny 52 / **small 112** / medium 107 / big 79 / xl 88) |
| **解码器输入** | DINOv2-large **patch grid**(16×16×1024,非池化) | — | 保空间才不糊;池化向量解码必鬼影(已证) |
| **解码器损失** | L1 + 0.5·MSE(纯 L1 即可) | — | 对抗(GAN)会掉色/幻觉高频,可读性反降 |
| **训练数据** | ~9k 帧(挖矿 ~200 ep)| — | 9k > 24k(small 解码器吃 24k 会**过度平滑**,112→67) |
| **milestone 选择** | **自适应**:KMeans-96 → 覆盖率 → Otsu 自动阈值 | 数据自适应 | 不固定 K;V2.4 思路 |
| **value bins** | **自适应**:bins = milestone 进度位置 `Pord` + 端点{0,1} | NB = #milestone + 2(无超参) | value↔milestone **构造上精确一一对应**,不再手调 NB |
| **value 读出频率** | 30Hz,DP 窗 `lam=80, medw=45` | — | 频率窗按 fps 标定(挖矿仍 3Hz 即够) |
| **代表图本身** | **簇中心 = 簇内 large-grid 平均 → small 解码器解码** | — | 平滑、去具体化的"可读质心原型"(不绑单 episode 颜色) |

**一句话**:`DINOv2-large 编码器 + small(0.92M)空间解码器 + patch-grid 输入 + L1 + 9k 数据 + 自适应 milestone + NB=61 value bins`。

---

## 支撑证据(消融结论)

1. **编码器**:small→base→large,簇中心结构 92→89→**112**,large 最优(`crave_scale_ablation.png`)。
2. **解码器倒 U**(固定 large+9k):tiny 52 / **small 112(峰)** / medium 107 / big 79 / xl 88(`crave_large_decoder_ladder.png`)。太小欠拟合(糊)、太大幻觉高频(噪)。
3. **规模无效于"平均质心"**:编码器 22M→300M、解码器 0.76M→12.8M(17×)、数据 9k→24k 全顶满(配置 I),单帧 medoid 冲到 234,但**簇中心始终 80–112**(`crave_scale_ablation.png` A–I)。→ 簇中心瓶颈是"平均输入 ill-posed",与规模无关。
4. **选型不能用单一锐度指标**:recon 锐度 / 特征余弦都被"细节/噪声"带偏 → **机制原则(大编码器+小解码器)+ 人眼**定夺,自动标量会误导。
5. **数据量**:value 质量 ~100–200 ep 饱和(`crave_mine_episode_sweep.png`),500ep 过量;30Hz 用于读出而非挖矿。
6. **NB 细化**:NB 21→61,ep2302 访问 milestone 10→18,value 台阶与 milestone 一一对应(不跳号)。

---

## 参考实现 / 演示

- **enc/dec 消融**:`train_scripts/kai/data/crave_decoder_scale_ablation.py`(配置 A–O)+ `crave_scale_aggregate.py`(出图)。
- **ep2302 30Hz 端到端演示**(本标准配置):`train_scripts/kai/data/crave_ep2302_30hz_decoded.py`
  - 输出:value 曲线(30Hz)+ milestone 随时间 + **每个 milestone 用簇中心解码图表示**;静态 `crave_ep2302_30hz_decoded.png` + 视频 `crave_ep2302_30hz_decoded.mp4`(左相机 / 右上 value 游标 / 右下解码质心,逐帧同步,已逐帧验证)。
- 编码器权重(离线):`/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-large`(HF/镜像被墙,按本地路径加载)。

## 适用范围与诚实边界

- **用途**:整洁、去具体化的"可读质心原型"——做 milestone 词表/示意、不绑某条 demo 的布料颜色。
- **不替代最近真实帧**:要**最清晰**的代表图,仍用 **最近真实帧(medoid)**(锐度 476 vs 解码质心 ~112)。
- **不要追"合成平均质心的高锐度"**:可形变布料的"平均"数学上 ill-posed,加规模/换损失都救不动;small 解码器给的是**平滑可读原型**,这已是该路线的最优点。
- 要"最清晰的单帧合成"可选 large+XL 解码器**取 medoid**(medoid 234),但那是美化 exemplar、有幻觉风险,非平均质心。
