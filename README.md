# 多频金融时序预测（Multi-Frequency Prediction）

本子项目是从主实验 `multi-frequency/` 裁剪出的**独立可运行**版本，用于股票未来走势三分类预测
（大幅上涨 / 震荡 / 大幅下跌，或 5 分类）。核心思路是：把每一只股票的对齐窗口组织成**日频、周频、
分时**三个频率的特征，再（可选）融合**指数宏观信号**，交给深度学习模型做分类。

> 本文档主要面向**师弟**：你之前做的是美股日数据研究。下面的"你需要准备什么"一节，
> 会明确告诉你：已有的日数据能直接用，还需要**补上分时数据**、**自己制作周数据**、**获取指数数据**，
> 以及每个环节可以参考/改造哪个脚本。

---

## 一、项目结构

```
multi_freq_prediction/
├── train.py                     # 训练 / 验证 / 测试 主脚本
├── utils/
│   ├── config.yaml              # 训练超参数 + 模型选择 + 数据路径
│   └── utils.py                 # 训练辅助工具（日志、指标、EarlyStop 等）
├── data/
│   ├── dataset.py               # 数据加载：读取 processeddata2 并构造 DataLoader
│   └── processeddata2/          # 处理后的张量数据（训练/验证/测试 .npy）
│       ├── train/  ├── val/  └── test/
└── models/
    ├── __init__.py              # get_model 工厂：按 model_type 返回模型
    ├── baseline_fusion_gate.py  # 主实验：市场状态门控三频融合（Gate）
    ├── baseline_fusion_gate_viz.py  # 可视化副本（权重与 Gate 一致）
    ├── baseline_ratio4idx.py    # Gate 的父类 / 4指数 ratio 编码
    ├── baseline_simple.py       # 基础分类头、序列编码器
    ├── seqnets.py               # TCN / GRU 等序列网络
    ├── loss.py                  # FocalLoss / SupConLoss 等
    ├── decomp.py  ssa.py        # 时序分解（SSA/FFT），可选
    └── tslib_models/            # ⭐ Time-Series-Library 经典模型（对比基线）
        ├── wrappers.py          # 对外 11 个 *Classifier 分类器
        ├── Transformer.py  Informer.py  Autoformer.py
        ├── PatchTST.py  TimesNet.py  iTransformer.py
        ├── TimeMixer.py
        ├── layers/              # 各模型的底层算子
        ├── moderntcn/  utils/
```

`data/dataset.py` 会自动读取 `processeddata2/{train,val,test}/` 下的 `.npy`，无需改代码；
数据文件的命名和形状见下文"数据格式规范"。

---

## 二、核心概念：三频特征 + 指数宏观

对每个样本（股票 × 决策日），取**以决策日结尾的最近 60 个交易日**为窗口，组织成：

| 频率           | 形状                                  | 含义                                                         |
| -------------- | ------------------------------------- | ------------------------------------------------------------ |
| 日频 daily     | `(60, 8)` → reshape `(12, 5, 8)` | 60 天日线：open/high/low/close/pct_chg/volume/turn/amplitude |
| 周频 weekly    | `(12, 8)` → `(1, 12, 8)`         | 把 60 天按每 5 天一组合成 12 周                              |
| 分时 intraday  | `(60, 8, 8)`                        | 每天取 8 根分时 bar                                          |
| 指数宏观 macro | `(60, 56)`                          | 4 个指数的日频 14 维特征 × 4（可选）                        |

模型通常接收 `(daily, weekly, intraday)`；fusion_gate 主实验额外接收 `macro`。

---

## 三、你需要准备什么（师弟必读）

你只有**美股日数据**。要让这套框架跑起来，需要补齐以下四类数据，并把它们组织成脚本能读的格式。
下面按**先后顺序**说明，每一步都告诉你"准备什么 + 参考哪个文件"。

### 第 1 步：日线数据（你已经有了）

**需要**：每只股票按 `code, date` 逐日一行，包含 8 列：
`open, high, low, close, pct_chg, volume, turn, amplitude`。

- 若你没有 `amplitude`（振幅）、`turn`（换手率），需自行计算：
  - `amplitude = (high - low) / prev_close * 100`
  - `turn = volume / 总股本 * 100`（见第 2 步的总股本）
- 参考：主项目 `data/rawdata/day/all_2016_2025_with_amplitude.csv` 的列结构，
  由脚本 `build_indices_new.py` 第 31 行 `DAILY_COLS` 定义。

### 第 2 步：分时数据（你**必须自己补**）

> 这是美股数据里最可能缺失的部分。分时特征是本框架区别于纯日线研究的关键。

**需要**：每只股票每天至少 8 根分时 bar（每根 bar 含：开盘/最高/最低/收盘/成交量）。

- 参考列名（脚本 `preprocess_data.py` 第 30 行 `INTRA_COLS`）：
  `开盘, 最高, 最低, 收盘, 成交量, 分时涨跌幅, 分时振幅, 分时换手率`
- `分时涨跌幅 / 分时振幅 / 分时换手率` 这三个衍生列，可直接参考脚本
  **`process_intraday_features.py`** 的 Step 2 计算逻辑：
  - 分时涨跌幅 =（当前收盘 − 昨日收盘）/ 昨日收盘 × 100
  - 分时振幅 =（当日截至目前最高 − 最低）/ 昨日收盘 × 100
  - 分时换手率 = 当日累计成交量 × 100 / 总股本 × 100
- 若你的分时数据粒度更细（如 1 分钟），需要**重采样**成 8 根 bar/天（或改脚本里的
  `N_BARS_PER_DAY = 8`）。
- 如果你**完全拿不到分时数据**，有两种办法：
  1. 把日线"降采样"模拟成分时（不推荐，信息量等同日线）；
  2. 修改脚本把分时维度退化（如每天 1 根 = 日线复制），并在 README 里注明这是简化版。

### 第 3 步：周数据（你**自己制作**）

本框架的周频特征不是单独下载的，而是**由日线合成**。

- 参考脚本 **`build_indices_new.py`** 的 Step 3：
  对每个 60 天窗口按每 5 天一组（共 12 组），计算 `open_avg/high_avg/low_avg/close_avg/volume_avg/cum_return/turn_avg/amplitude_avg`。
- 周收益 `cum_return = (last_close − first_close) / first_close`。
- 你只要确保日线数据完整，`build_indices_new.py` 就能自动生成周特征，**无需额外下载周线**。

### 第 4 步：指数数据（你**需要自己获取**）

主实验的 fusion_gate 模型可选地使用"4 个指数"的宏观信号（如沪深300、中证500）。
**换成美股后，你需要选 4 个代表美股市场的指数**（例如标普500、纳斯达克100、道琼斯、罗素2000），
并获取它们每天的：`open/high/low/close/volume/turn/pct_chg` 以及上涨/下跌家数等。

- 参考脚本 **`preprocess_macro_new.py`**：
  - 从 Excel（或 csv）读 4 个指数，每个指数计算 14 维特征 → 4 指数拼成 56 维。
  - 这 56 维会写入 `macro4idx.npy` / `macro4idx_weekly.npy`。
- 若你暂时**不做指数宏观**，最简单的方式：把 `config.yaml` 中 `model.type` 换成
  不依赖 macro 的模型（如 `baseline_transformer` / `baseline_timemixer` 等），
  `data/dataset.py` 在没有 `macro4idx.npy` 时会自动返回 `macro=None`，不影响运行。

### 第 5 步：标签（分类目标）

训练需要每个样本的**未来标签**。参考主项目 `future_50d_close.csv`，其中每条记录包含
`code, date` 及辅助字段。标签通常由未来 50 天收益决定：

- 大幅上涨 / 小幅上涨 / 震荡 / 小幅下跌 / 大幅下跌（5 类）
- 框架在 `num_classes=3` 时会自动把 5 类映射成 3 类：
  `{1→0, 2→1, 3/4→2}`（见 `dataset.py` 的映射逻辑）。
- 你还需要为每个样本生成 `label_window_index_new.csv`（窗口起止行号）——由
  `build_indices_new.py` 生成。

---

## 四、数据处理脚本（4 个，按顺序）

主项目 `multi-frequency/` 下的 4 个脚本是数据处理的核心。**它们的产出就是本子项目
`processeddata2/` 的输入**。师弟需要把这 4 个脚本的输入换成美股数据后重新跑一遍。

| 顺序 | 脚本                             | 作用                                                      | 关键产出                                                                                       | 必须前置              |
| ---- | -------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | --------------------- |
| ①   | `process_intraday_features.py` | 推算总股本、重算分时特征、重建分时对齐索引（独立/备用线） | `total_shares.csv`, `intraday_sample_index.csv`                                            | 分时原始数据          |
| ②   | `build_indices_new.py`         | 基于标签文件生成窗口索引 / 周频特征 / 分时对齐索引        | `label_window_index_new.csv`, `weekly_features_new.csv`, `intraday_sample_index_new.csv` | 标签 + 日线           |
| ③   | `preprocess_data.py`           | 生成训练/验证/测试张量`processeddata2/{split}/*.npy`    | `daily/weekly/intraday/labels` 等 7 个 `.npy`                                              | **必须先跑 ②** |
| ④   | `preprocess_macro_new.py`      | 对齐 4 指数宏观数据                                       | `macro4idx.npy`, `macro4idx_weekly.npy`                                                    | **必须先跑 ③** |

### 重要顺序约定

1. **`build_indices_new.py` 必须先跑**：它产出 ③④ 都依赖的 `label_window_index_new.csv`。
2. **`preprocess_macro_new.py` 必须在 `preprocess_data.py` 之后跑**：两者用**同一套切分逻辑**
   （`seed=42`、分层 `7:1:2`、最终排序），保证股票特征与指数宏观逐样本对齐；macro 脚本还会读取
   `daily.npy` 校验样本数。
3. `process_intraday_features.py` 相对独立，主要在校准/更新分时原始数据质量时用。

### 脚本各自输入/输出明细

**`process_intraday_features.py`**

- 输入：`data/rawdata/day/*.csv`、`data/rawdata/intraday/*.csv`、`classification_labeled.csv`、`label_window_index.csv`、可选 `30分钟/`
- 输出：`total_shares.csv`（总股本）、重写 `intraday/*.csv`、`intraday_sample_index.csv`、`intraday_date_lookup.csv`、`missing_intraday_samples.csv`
- 运行：`python process_intraday_features.py`（无参数）

**`build_indices_new.py`sika**

- 输入：`future_50d_close.csv`（标签）、`day/all_*.csv`（日线）、`intraday_sample_index.csv`（旧分时索引，可选）
- 输出：`label_window_index_new.csv`、`weekly_features_new.csv`、`intraday_sample_index_new.csv`
- 运行：`python build_indices_new.py`

**`preprocess_data.py`**

- 输入：`future_50d_close.csv`、上面 ② 的 3 个 `_new.csv`、日线、分时原始文件
- 输出（`data/processeddata2/{train,val,test}/`）：| 文件                   | 形状           | 含义                               |
  | ---------------------- | -------------- | ---------------------------------- |
  | `labels.npy`         | `(N,)`       | 分类标签                           |
  | `daily.npy`          | `(N,12,5,8)` | 日频                               |
  | `weekly.npy`         | `(N,1,12,8)` | 周频                               |
  | `intraday.npy`       | `(N,60,8,8)` | 分时                               |
  | `future_ret_seq.npy` | `(N,50)`     | 未来 50 天逐日收益率（多任务可选） |
  | `future_cum_seq.npy` | `(N,50)`     | 未来 50 天累计收益曲线             |
  | `aux_stats.npy`      | `(N,8)`      | 8 维辅助统计量                     |
- 运行：`python preprocess_data.py`

**`preprocess_macro_new.py`**

- 输入：`label_window_index_new.csv`、`future_50d_close.csv`、指数 Excel（如 `index_300_500_1000_all.xlsx`）、`processeddata2/{split}/daily.npy`
- 输出：`macro4idx.npy` `(N,60,56)`、`macro4idx_weekly.npy` `(N,12,56)`
- 运行：`python preprocess_macro_new.py [--processed-dir DIR] [--excel FILE]`

---

## 五、数据格式规范（processeddata2）

`data/dataset.py` 按如下约定读取。师弟只要把美股数据按这个格式组织好即可。

- 目录：`processeddata2/train/`, `processeddata2/val/`, `processeddata2/test/`
- 每个目录下放 `.npy` 文件（上面表格那 7 个；`macro4idx*.npy` 可选）
- 训练/验证/测试划分比例：**7 : 1 : 2**（`seed=42` 分层）
- 所有 split 的 `labels` 与各特征数组第一维长度必须一致（`dataset.py` 会自动取最短长度兜底，
  但请务必保证一致，否则会丢弃样本）

**3 类标签映射**（`num_classes=3` 时）：5 类原始标签 `{1,2,3,4}`（假设 `0` 为其他）会映射为：

```
原始: 1 → 0    2 → 1    3 → 2    4 → 2
```

---

## 六、训练与模型选择

### 训练

```bash
python train.py --config utils/config.yaml
```

`utils/config.yaml` 中关键项：

- `model.type`：选择模型（见下表）
- `data.processed_dir`：数据目录，默认 `data/processeddata2`
- `data.macro4idx_suffix`：留空即加载 `macro4idx.npy`
- `training.epochs / batch_size / learning_rate` 等

### 可用模型（`model.type`）

**主实验（门控融合，推荐）**

| model.type                                    | 说明                                           |
| --------------------------------------------- | ---------------------------------------------- |
| `scale_centric_fusion_gate_mha`             | ⭐ 市场状态门控 + MHA 市场编码（主实验，默认） |
| `scale_centric_fusion_gate_factor`          | 门控 + Factor 市场编码                         |
| `scale_centric_fusion_gate_softmax`         | 门控 + Softmax 市场编码                        |
| `scale_centric_fusion_gate_relation_matrix` | 门控 + 关系矩阵市场编码                        |
| `scale_centric_fusion_gate_mha_viz`         | 可视化副本（权重与 mha 一致）                  |

加权融合（`scale_centric_fusion_weighted_*`）与 FiLM 融合（`scale_centric_fusion_film_*`）同构可选。

**对比基线（来自 `tslib_models`，无市场信息）**

| model.type                           | 模型         |
| ------------------------------------ | ------------ |
| `baseline_lstm` / `baseline_gru` | LSTM / GRU   |
| `baseline_tcn`                     | 时序卷积 TCN |
| `baseline_transformer`             | Transformer  |
| `baseline_informer`                | Informer     |
| `baseline_autoformer`              | Autoformer   |
| `baseline_patchtst`                | PatchTST     |
| `baseline_timesnet`                | TimesNet     |
| `baseline_itransformer`            | iTransformer |
| `baseline_timemixer`               | TimeMixer    |
| `baseline_moderntcn`               | ModernTCN    |

这些 `tslib_models` 分类器均由 `models/tslib_models/wrappers.py` 统一封装，接收三频输入
`(daily, weekly, intraday)`，`get_model` 会自动按需传参。

---

## 七、依赖

核心依赖：`torch`、`numpy`、`pandas`、`pyyaml`、`tqdm`。
`tslib_models` 部分模型额外需要：`einops`、`scipy`、`matplotlib`。
可选（仅用到对应模型时才需要）：`sympy`、`PyWavelets`（DWT）、`mamba-ssm`（Mamba）、`reformer-pytorch`。

安装：

```bash
pip install torch numpy pandas pyyaml tqdm einops scipy matplotlib
```

---

## 八、给师弟的最终检查清单

- [X] 日线：每只股票含 `open/high/low/close/pct_chg/volume/turn/amplitude` 8 列
- [X] 分时：每只股票每天 ≥8 根 bar，含 价格 + 成交量（参考 `process_intraday_features.py` 重算衍生列）
- [X] 周频：**不用下载**，由 `build_indices_new.py` 从日线自动合成
- [X] 指数：选取 4 个美股市场指数，按 `preprocess_macro_new.py` 格式组织（可选，不做就用无 macro 模型）
- [X] 标签：生成 `future_50d_close.csv`（含 `code, date` 与未来收益相关字段）
- [X] 依次运行：`build_indices_new.py` → `preprocess_data.py` → `preprocess_macro_new.py`
- [X] 产出 `processeddata2/{train,val,test}/*.npy` 后，直接 `python train.py --config utils/config.yaml`

有任何一步卡住，重点看对应脚本头部 docstring（都有详细的列名和算法说明）。
