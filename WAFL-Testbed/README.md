# WAFL-Testbed

**日本語** | [English](#english-version)

---

## 日本語版

### 概要

WAFL-Testbed（物理・コンテナハイブリッドテストベッド）は，大規模実環境無線アドホック連合学習（WAFL: Wireless Ad-hoc Federated Learning）における同期方式・通信プロトコルのスケーラビリティと信頼性を実機ベースで評価するための研究プラットフォームである．

従来のシミュレーションベース研究では見えなかった**実環境固有の制約**（OS・ネットワークスタックの遅延，物理リソース競合，同期待ち時間の指数関数的増大）を定量化し，実用的な設計指針を提供する．

### 主要機能

#### 3 つの通信モード

WAFL-Testbed は用途に応じて 3 つの通信モードを提供する．RUDP プロトコルは廃止され，現在は TCP，UDP，および Fast モードのみをサポートする．

| モード   | 説明                                                                       | 適用環境                 |
| -------- | -------------------------------------------------------------------------- | ------------------------ |
| **TCP**  | 標準 TCP 接続による信頼性重視の通信．OS のスタックをそのまま使用する．     | 安定したネットワーク環境 |
| **UDP**  | UDP + FEC による適応的通信．ネットワーク推定に基づき冗長度を自動調整する． | 不安定なネットワーク環境 |
| **Fast** | UDP + FEC による高速通信．高帯域環境向けに最適化されたパラメータを使用．   | 高帯域・低損失環境       |

詳細: [docs/protocol.md](docs/protocol.md)

#### Semi-Synchronous Protocol (SSP) - Autonomous

従来の中央集権型 SSP ではなく，各実行サーバが自律的に同期を制御する分散型 SSP を実装している．
- 各実行サーバがモデル交換時に自律的に SSP 制御を実行する．
- 閾値 `ssp_threshold` で完了率を制御する（例: 0.8 = 80 % のピアが完了した時点で残りをキャンセルする）．
- 管理サーバは SSP 設定の共有のみを行い，実際の同期タイミング制御には関与しない．
- 完了したピアから順次集約を開始するため，ストラグラー（遅延ノード）による全体的な停滞を回避できる．
- 破棄された計算量の詳細メトリクス（`wasted_ms`，`wasted_norm`，`batches_processed`）を記録する．

#### UDP + XOR-based FEC (Forward Error Correction)

パケットロスが発生しやすい無線環境において，再送による遅延を回避するために FEC を導入している．
- zfec ライブラリによる Block-based XOR 冗長パケットを生成する．
- $k$ 個のデータパケットに対して $m$ 個のパリティパケットを付加し，任意の $k$ 個を受信できれば復元可能である．
- ネットワーク品質（損失率）に基づく適応的 FEC 冗長度調整（UDP モード）を行う．
- パケット受信の停滞を検知して早期に再送を要求する Proactive NACK 機構を備える．

#### Adaptive Compression

通信帯域と CPU 負荷のトレードオフを考慮し，動的に圧縮方式を選択する．
- サポート方式: None，LZ4 (高速)，zlib (高圧縮)
- 数理モデルに基づく最適化: $T_{est} = T_{comp} + (Size_{comp} \times R) / BW$
- ここで $R$ は FEC 冗長率，$BW$ は推定帯域，$T_{comp}$ は圧縮にかかる予測時間である．

#### Mobility-Aware Network Emulation

移動体通信環境をシミュレートするため，SUMO と Linux の制御機能を統合している．
- SUMO シミュレーションからのモビリティトレースをパースする．
- ノード間距離に基づく動的ネットワーク品質エミュレーションを行う．
- HTB (Hierarchical Token Bucket) とフィルタを用いて，宛先（ピア）ごとに独立した帯域制限・遅延・損失を適用する．
- 4 段階の品質ランク（Excellent/Good/Fair/Poor）に基づき，パラメータを動的に変更する．
- 詳細: [docs/mobility_aware.md](docs/mobility_aware.md)

#### トポロジー生成

| モデル                           | 説明                                                         | ツール                           |
| -------------------------------- | ------------------------------------------------------------ | -------------------------------- |
| **Random Waypoint (RWP)**        | ノードが一定速度で移動し続ける動的シナリオ用                 | `utils/generate_rwp_topology.py` |
| **Random Geometric Graph (RGG)** | 位置固定の静的トポロジー（Dense: 平均次数 ≥ 10 / Sparse: 4） | `utils/generate_rgg_topology.py` |
| **SUMO Mobility Trace**          | 実地図データに基づくリアルな移動パターン                     | `mise sumo` / `mise sumo-osm`    |

### クイックスタート

#### 1. 前提条件

**コントロールサーバー**:
- OS: Linux（Ubuntu 推奨）
- Python 3.11+
- [mise](https://mise.jdx.dev/)（タスクランナー）

**実行サーバー（エージェント）**:
- OS: Linux
- Docker（インストール済み＆実行中）

**ネットワーク**:
- コントロールサーバーから全実行サーバーへのパスワードなし SSH アクセス

#### 2. セットアップ

```bash
# 1. mise インストール
curl https://mise.run | sh

# 2. シェルで mise を有効化（bash の例）
echo 'eval "$(~/.local/bin/mise activate bash)"' >> ~/.bashrc
source ~/.bashrc

# 3. プロジェクトディレクトリで依存関係インストール
cd WAFL-Testbed
mise setup
```

#### 3. SSH 設定

```bash
# 1. SSH キーペア生成（既存のものがない場合）
ssh-keygen -t ed25519

# 2. 公開鍵を各実行サーバーにコピー
ssh-copy-id denjo@192.168.11.100
ssh-copy-id denjo@192.168.11.101
# ... 全ノードで繰り返し
```

#### 4. 設定ファイル編集

**`ctrl/execution_config.json`** - インフラ設定:
```json
{
  "nodes": [
    {
      "name": 0,
      "physical_ip": "192.168.11.100",
      "container_port_ctrl": 10001,
      "host_port_ctrl": 10001,
      "host_port_p2p": 10002,
      "cpu_limit": "1.0"
    }
  ],
  "deployment_location": "/home/denjo",
  "user": "denjo"
}
```

**`ctrl/parameters.json`** - 実験パラメータ:
```json
{
  "experiment_name": "My Experiment",
  "epochs": {"self": 64, "wafl": 4096},
  "contact_pattern": "contact_pattern/rwp_n28_a1000_r100_p10_s01.json",
  "wafl_phase": {
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "network_condition": {
    "enabled": true,
    "rate": "10mbit",
    "delay": "50ms",
    "loss": "3%"
  },
  "method": "udp",
  "ssp": {
    "enabled": true,
    "ssp_threshold": 0.8
  }
}
```

#### 5. 実験実行

```bash
# トポロジー生成（初回のみ）
python utils/generate_rgg_topology.py --nodes 3 --epochs 100 --density dense

# 全ノードへデプロイ＆実験開始
mise start

# 結果分析（実験終了後）
mise analyze

# 停止: Ctrl+C（全エージェントが正常終了）
```

### mise タスク一覧

| タスク          | コマンド           | 説明                                     |
| --------------- | ------------------ | ---------------------------------------- |
| **rsync**       | `mise rsync`       | プロジェクトファイルを管理サーバーに同期 |
| **setup**       | `mise setup`       | 依存関係インストール＆ノードセットアップ |
| **deploy**      | `mise deploy`      | Docker イメージビルド＆全ノードへ配布    |
| **start**       | `mise start`       | デプロイ後に実験を開始                   |
| **experiments** | `mise experiments` | `ctrl/parameters/` 内の全実験を順次実行  |
| **analyze**     | `mise analyze`     | 結果収集＆分析＆グラフ生成               |
| **verify**      | `mise verify`      | 設定検証＆ベンチマーク実行               |
| **sumo**        | `mise sumo`        | SUMO モビリティ前処理パイプライン        |
| **sumo-osm**    | `mise sumo-osm`    | OSM 実地図モードでの SUMO 前処理         |

### ディレクトリ構造

```
WAFL-Testbed/
├── docs/                   # 詳細ドキュメント
│   ├── architecture.md     # システムアーキテクチャ
│   ├── configuration.md    # 設定ガイド
│   ├── setup.md            # セットアップガイド
│   ├── usage.md            # 使用方法
│   ├── protocol.md         # 通信プロトコル詳細
│   ├── analysis.md         # 結果分析ガイド
│   └── mobility_aware.md   # モビリティ対応機能
├── ctrl/                   # コントロールサーバー
│   ├── main.py             # オーケストレーター
│   ├── deploy.py           # デプロイスクリプト
│   ├── analyze.py          # 結果分析・グラフ生成
│   ├── verify.py           # 設定検証・ベンチマーク
│   ├── run_experiments.py  # 複数実験の自動実行
│   ├── execution_config.json  # インフラ設定
│   ├── parameters.json     # 実験パラメータ
│   └── parameters/         # 複数実験パラメータファイル
├── wafl/                   # 実行エージェント
│   ├── src/common/
│   │   ├── main.py         # エージェントメイン
│   │   ├── udp_model_sharing.py  # UDP / FEC 実装
│   │   ├── compression_manager.py  # Adaptive Compression
│   │   ├── network_estimator.py    # ネットワーク状態推定
│   │   └── logger.py       # 構造化ログ
│   └── config/             # エージェント設定（自動生成）
├── utils/                  # ユーティリティ
│   ├── generate_rwp_topology.py     # RWP トポロジー生成
│   ├── generate_rgg_topology.py     # RGG トポロジー生成
│   ├── prepare_mobility.py          # SUMO モビリティ前処理
│   ├── visualize_sumo_results.py    # SUMO 結果可視化
│   └── generate_datasets.py         # データセット生成
├── data/                   # データ・トポロジー
│   ├── contact_pattern/    # 接触パターン JSON
│   └── sumo/               # SUMO モビリティデータ
└── results/                # 実験結果
```

### ドキュメント

| ドキュメント                                   | 内容                              |
| ---------------------------------------------- | --------------------------------- |
| [システムアーキテクチャ](docs/architecture.md) | 設計とコンポーネント              |
| [セットアップガイド](docs/setup.md)            | インストールと初期設定            |
| [設定ガイド](docs/configuration.md)            | パラメータ詳細                    |
| [使用方法](docs/usage.md)                      | 実験実行手順                      |
| [通信プロトコル詳細](docs/protocol.md)         | TCP / UDP / Fast モードの実装詳細 |
| [結果分析ガイド](docs/analysis.md)             | グラフ・レポートの解釈方法        |
| [モビリティ対応機能](docs/mobility_aware.md)   | SUMO 統合と動的ネットワーク制御   |

### ログとメトリクス

実験結果は構造化された JSON Lines 形式で記録される：

```jsonl
{"timestamp": 1732567890.123, "node": "0", "type": "epoch_complete", "epoch": 1, "train_acc": 0.95, ...}
{"timestamp": 1732567890.456, "node": "0", "type": "ssp_force_next", "wasted_ms": 1234.56, ...}
{"timestamp": 1732567890.789, "node": "0", "type": "udp_stats", "survival_rate": 0.98, ...}
```

**主要メトリクス**:
- **効率性**: Wall-clock time, Wasted Computation
- **ネットワーク**: Goodput, Survival Rate, Traffic Volume
- **システム負荷**: CPU Usage, NIC Usage
- **学習品質**: Accuracy vs Time

詳細: [docs/analysis.md](docs/analysis.md)

---

## English Version

### Overview

WAFL-Testbed is a research platform for evaluating the **scalability and reliability of synchronization schemes and communication protocols** in large-scale, real-world Wireless Ad-hoc Federated Learning (WAFL) environments using a hybrid physical/container infrastructure.

Unlike traditional simulation-based research, this testbed quantifies **real-world constraints** invisible in simulators: OS/network stack latency, physical resource contention, and exponential synchronization overhead growth.

### Key Features

#### Three Communication Modes

WAFL-Testbed provides three communication modes for different use cases:

| Mode     | Description                                                                             | Use Case                              |
| -------- | --------------------------------------------------------------------------------------- | ------------------------------------- |
| **TCP**  | Model exchange via standard TCP connections with full OS network stack                  | Stable network environments           |
| **UDP**  | Adaptive UDP + FEC communication with parameter auto-tuning based on network estimation | Unstable network environments         |
| **Fast** | High-speed UDP + FEC communication optimized for high-bandwidth paths                   | High-bandwidth, low-loss environments |

Details: [docs/protocol.md](docs/protocol.md)

#### Semi-Synchronous Protocol (SSP) - Autonomous

- Each execution server autonomously manages SSP during model exchange
- Configurable threshold `ssp_threshold` (e.g., 0.8 = cancel remaining exchanges when 80% of peers complete)
- Control server only shares SSP configuration, no control involvement
- Detailed wasted computation metrics (`wasted_ms`, `wasted_norm`, `batches_processed`)

#### UDP + XOR-based FEC (Forward Error Correction)

- Avoids TCP retransmission delays with fast UDP transfer
- Block-based XOR redundancy using zfec library
- Adaptive FEC redundancy based on network quality
- Efficient packet loss recovery via Proactive NACK
- Survival rate and FEC recovery statistics tracking

#### Adaptive Compression

- Dynamic compression method selection based on bandwidth and CPU load
- Supported methods: None, LZ4 (fast), zlib (high compression)
- Measurement-based optimization: $T_{est} = T_{comp} + (Size_{comp} \times R) / BW$

#### Mobility-Aware Network Emulation

- SUMO simulation-based mobility trace generation
- Distance-based dynamic network quality emulation
- Per-Peer Limitation: HTB + Filter for per-destination network constraints
- 4-tier quality ranks (Excellent/Good/Fair/Poor)
- Details: [docs/mobility_aware.md](docs/mobility_aware.md)

#### Topology Generation

| Model                            | Description                                                     | Tool                             |
| -------------------------------- | --------------------------------------------------------------- | -------------------------------- |
| **Random Waypoint (RWP)**        | Standard WAFL scenario with continuous node movement            | `utils/generate_rwp_topology.py` |
| **Random Geometric Graph (RGG)** | Static topology (Dense: avg degree ≥10 / Sparse: avg degree ≤4) | `utils/generate_rgg_topology.py` |
| **SUMO Mobility Trace**          | Realistic movement patterns from SUMO simulations               | `mise sumo`                      |

### Quick Start

#### 1. Prerequisites

**Control Server**:
- OS: Linux (Ubuntu recommended)
- Python 3.11+
- [mise](https://mise.jdx.dev/) (task runner)

**Execution Servers (Agents)**:
- OS: Linux
- Docker (installed & running)

**Network**:
- Passwordless SSH access from Control Server to all Execution Servers

#### 2. Setup

```bash
# 1. Install mise
curl https://mise.run | sh

# 2. Activate mise in shell (bash example)
echo 'eval "$(~/.local/bin/mise activate bash)"' >> ~/.bashrc
source ~/.bashrc

# 3. Install dependencies in project directory
cd WAFL-Testbed
mise setup
```

#### 3. SSH Configuration

```bash
# 1. Generate SSH key pair (if you don't have one)
ssh-keygen -t ed25519

# 2. Copy public key to each Execution Server
ssh-copy-id denjo@192.168.11.100
ssh-copy-id denjo@192.168.11.101
# ... repeat for all nodes
```

#### 4. Edit Configuration Files

**`ctrl/execution_config.json`** - Infrastructure:
```json
{
  "nodes": [
    {
      "name": 0,
      "physical_ip": "192.168.11.100",
      "container_port_ctrl": 10001,
      "host_port_ctrl": 10001,
      "host_port_p2p": 10002,
      "cpu_limit": "1.0"
    }
  ],
  "deployment_location": "/home/denjo",
  "user": "denjo"
}
```

**`ctrl/parameters.json`** - Experiment Parameters:
```json
{
  "experiment_name": "My Experiment",
  "epochs": {"self": 64, "wafl": 4096},
  "contact_pattern": "contact_pattern/rwp_n28_a1000_r100_p10_s01.json",
  "wafl_phase": {
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "network_condition": {
    "enabled": true,
    "rate": "10mbit",
    "delay": "50ms",
    "loss": "3%"
  },
  "method": "udp",
  "ssp": {
    "enabled": true,
    "ssp_threshold": 0.8
  }
}
```

#### 5. Run Experiment

```bash
# Generate topology (first time only)
python utils/generate_rgg_topology.py --nodes 3 --epochs 100 --density dense

# Deploy to all nodes and start experiment
mise start

# Analyze results (after experiment completion)
mise analyze

# Stop: Press Ctrl+C (graceful shutdown of all agents)
```

### mise Task List

| Task            | Command            | Description                                            |
| --------------- | ------------------ | ------------------------------------------------------ |
| **rsync**       | `mise rsync`       | Sync project files to management server                |
| **setup**       | `mise setup`       | Install dependencies & setup nodes                     |
| **deploy**      | `mise deploy`      | Build Docker image & distribute to all nodes           |
| **start**       | `mise start`       | Start experiment after deployment                      |
| **experiments** | `mise experiments` | Run all experiments in `ctrl/parameters/` sequentially |
| **analyze**     | `mise analyze`     | Collect results, analyze & generate graphs             |
| **verify**      | `mise verify`      | Run configuration verification & benchmarks            |
| **sumo**        | `mise sumo`        | SUMO mobility preprocessing pipeline                   |
| **sumo-osm**    | `mise sumo-osm`    | SUMO preprocessing with OSM real-world maps            |

### Directory Structure

```
WAFL-Testbed/
├── docs/                   # Detailed documentation
│   ├── architecture.md     # System architecture
│   ├── configuration.md    # Configuration guide
│   ├── setup.md            # Setup guide
│   ├── usage.md            # Usage guide
│   ├── protocol.md         # Communication protocol details
│   ├── analysis.md         # Results analysis guide
│   └── mobility_aware.md   # Mobility-aware features
├── ctrl/                   # Control Server (Management)
│   ├── main.py             # Server main orchestrator
│   ├── deploy.py           # Deployment agent (SSH/Docker)
│   ├── analyze.py          # Results analysis & graph generation
│   ├── verify.py           # Configuration verification & benchmarks
│   ├── run_experiments.py  # Automated multi-experiment execution
│   ├── execution_config.json  # Infrastructure config
│   ├── parameters.json     # Experiment parameters
│   └── parameters/         # Multiple experiment parameter files
├── wafl/                   # Execution Agents
│   ├── src/common/
│   │   ├── main.py         # Agent main
│   │   ├── udp_model_sharing.py  # UDP / FEC implementation
│   │   ├── compression_manager.py  # Adaptive Compression
│   │   ├── network_estimator.py    # Network state estimation
│   │   └── logger.py       # Structured logging
│   └── config/             # Agent config (auto-generated)
├── utils/                  # Utilities
│   ├── generate_rwp_topology.py     # RWP topology generation
│   ├── generate_rgg_topology.py     # RGG topology generation
│   ├── prepare_mobility.py          # SUMO mobility preprocessing
│   ├── visualize_sumo_results.py    # SUMO results visualization
│   └── generate_datasets.py         # Dataset generation
├── data/                   # Data & Topology
│   ├── contact_pattern/    # Contact pattern JSONs
│   └── sumo/               # SUMO mobility data
└── results/                # Experiment results
```

### Documentation

| Document                                           | Content                                      |
| -------------------------------------------------- | -------------------------------------------- |
| [System Architecture](docs/architecture.md)        | Design and components                        |
| [Setup Guide](docs/setup.md)                       | Installation and initial setup               |
| [Configuration Guide](docs/configuration.md)       | Parameter details                            |
| [Usage Guide](docs/usage.md)                       | Experiment execution                         |
| [Communication Protocol Details](docs/protocol.md) | TCP / UDP / Fast mode implementation details |
| [Results Analysis Guide](docs/analysis.md)         | Graph and report interpretation              |
| [Mobility-Aware Features](docs/mobility_aware.md)  | SUMO integration and dynamic network control |

### Logs and Metrics

Experiment results are recorded in structured JSON Lines format:

```jsonl
{"timestamp": 1732567890.123, "node": "0", "type": "epoch_complete", "epoch": 1, "train_acc": 0.95, ...}
{"timestamp": 1732567890.456, "node": "0", "type": "ssp_force_next", "wasted_ms": 1234.56, ...}
{"timestamp": 1732567890.789, "node": "0", "type": "udp_stats", "survival_rate": 0.98, ...}
```

**Key Metrics**:
- **Efficiency**: Wall-clock time, Wasted Computation
- **Network**: Goodput, Survival Rate, Traffic Volume
- **System Load**: CPU Usage, NIC Usage
- **Learning Quality**: Accuracy vs Time

Details: [docs/analysis.md](docs/analysis.md)
