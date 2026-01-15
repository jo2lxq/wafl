# WAFL-Testbed

**日本語** | [English](#english-version)

---

## 日本語版

### 概要

WAFL-Testbed（物理・コンテナハイブリッドテストベッド）は，大規模実環境無線アドホック連合学習（WAFL: Wireless Ad-hoc Federated Learning）における同期方式・通信プロトコルのスケーラビリティと信頼性を実機ベースで評価するための研究プラットフォームである．

従来のシミュレーションベース研究では見えなかった**実環境固有の制約**（OS・ネットワークスタックの遅延，物理リソース競合，同期待ち時間の指数関数的増大）を定量化し，実用的な設計指針を提供する．

### 主要機能

#### 3 つの通信モード

WAFL-Testbed は用途に応じて 3 つの通信モードを提供する：

| モード      | 説明                                                                   | 適用環境                 |
| ----------- | ---------------------------------------------------------------------- | ------------------------ |
| **TCP**     | 標準 TCP 接続による信頼性重視の通信                                    | 安定したネットワーク環境 |
| **Dynamic** | UDP+FEC による適応的通信（ネットワーク推定に基づくパラメータ自動調整） | 不安定なネットワーク環境 |
| **Fast**    | UDP+FEC による高速通信（高帯域環境向け最適化）                         | 高帯域・低損失環境       |

詳細: [docs/protocol.md](docs/protocol.md)

#### Semi-Synchronous Protocol (SSP) - Reset Model

- 遅延ノードを切り捨てて学習速度を優先
- 閾値 `ssp_threshold` で完了率を制御（例: 0.8 = 80% 完了で強制進行）
- 1 エポック以上遅れるノードは存在しない設計
- 破棄された計算量の詳細メトリクス（`wasted_ms`, `wasted_norm`, `batches_processed`）

#### UDP + XOR-based FEC (Forward Error Correction)

- TCP 再送遅延を回避した高速 UDP 通信
- zfec ライブラリによる Block-based XOR 冗長パケット
- ネットワーク品質に基づく適応的 FEC 冗長度調整
- Proactive NACK による効率的なパケットロス回復
- 生存率 (Survival Rate) と FEC 復元統計の記録

#### Adaptive Compression

- 帯域と計算負荷に応じた動的圧縮方式選択
- サポート方式: None, LZ4 (高速), zlib (高圧縮)
- 実測ベースの最適化: $T_{est} = T_{comp} + (Size_{comp} \times R) / BW$

#### Mobility-Aware Network Emulation

- SUMO シミュレーションによるモビリティトレース生成
- ノード間距離に基づく動的ネットワーク品質エミュレーション
- Per-Peer Limitation: HTB + Filter による相手ごとのネットワーク制限
- 4 段階品質ランク（Excellent/Good/Fair/Poor）
- 詳細: [docs/mobility_aware.md](docs/mobility_aware.md)

#### トポロジー生成

| モデル                           | 説明                                                      | ツール                           |
| -------------------------------- | --------------------------------------------------------- | -------------------------------- |
| **Random Waypoint (RWP)**        | ノードが移動し続ける標準的な WAFL シナリオ                | `utils/generate_rwp_topology.py` |
| **Random Geometric Graph (RGG)** | 静的トポロジー（Dense: 平均次数≥10 / Sparse: 平均次数≤4） | `utils/generate_rgg_topology.py` |
| **SUMO Mobility Trace**          | SUMO シミュレーションからのリアルな移動パターン           | `mise sumo`                      |

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
  "method": "dynamic",
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
│   │   ├── udp_model_sharing.py  # UDP/FEC 実装
│   │   ├── rudp_protocol.py      # RUDP プロトコル
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
| [通信プロトコル詳細](docs/protocol.md)         | TCP/Dynamic/Fast モードの実装詳細 |
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

| Mode        | Description                                                                                | Use Case                              |
| ----------- | ------------------------------------------------------------------------------------------ | ------------------------------------- |
| **TCP**     | Reliable communication via standard TCP connections                                        | Stable network environments           |
| **Dynamic** | Adaptive UDP+FEC communication with automatic parameter tuning based on network estimation | Unstable network environments         |
| **Fast**    | High-speed UDP+FEC communication optimized for high-bandwidth environments                 | High-bandwidth, low-loss environments |

Details: [docs/protocol.md](docs/protocol.md)

#### Semi-Synchronous Protocol (SSP) - Reset Model

- Prioritizes learning speed by discarding slow nodes
- Configurable threshold `ssp_threshold` (e.g., 0.8 = force progress at 80% completion)
- Design ensures no node ever falls more than 1 epoch behind
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
  "method": "dynamic",
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
├── ctrl/                   # Control Server
│   ├── main.py             # Orchestrator
│   ├── deploy.py           # Deployment script
│   ├── analyze.py          # Results analysis & graph generation
│   ├── verify.py           # Configuration verification & benchmarks
│   ├── run_experiments.py  # Automated multi-experiment execution
│   ├── execution_config.json  # Infrastructure config
│   ├── parameters.json     # Experiment parameters
│   └── parameters/         # Multiple experiment parameter files
├── wafl/                   # Execution Agents
│   ├── src/common/
│   │   ├── main.py         # Agent main
│   │   ├── udp_model_sharing.py  # UDP/FEC implementation
│   │   ├── rudp_protocol.py      # RUDP protocol
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
| [Communication Protocol Details](docs/protocol.md) | TCP/Dynamic/Fast mode implementation details |
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
