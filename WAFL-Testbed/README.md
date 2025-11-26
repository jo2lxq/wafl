# WAFL-Testbed

**日本語** | [English](#english-version)

---

## 日本語版

### 概要

WAFL-Testbed（物理・コンテナハイブリッドテストベッド）は，大規模実環境無線アドホック連合学習（WAFL: Wireless Ad-hoc Federated Learning）における同期方式・通信プロトコルのスケーラビリティと信頼性を実機ベースで評価するための研究プラットフォームである．

従来のシミュレーションベース研究では見えなかった**実環境固有の制約**（OS・ネットワークスタックの遅延，物理リソース競合，同期待ち時間の指数関数的増大）を定量化し，実用的な設計指針を提供する．

### 主要機能

#### 実装済みプロトコル・手法

1. **Semi-Synchronous Protocol (SSP) - Reset Model**
   - 遅延ノードを切り捨てて学習速度を優先
   - 閾値 `ssp_threshold` で完了率を制御（例: 0.9 = 90% 完了で強制進行）
   - 破棄された計算量の詳細メトリクス（`wasted_ms`, `wasted_norm`, `batches_processed`）

2. **UDP + XOR-based FEC (Forward Error Correction)**
   - TCP 再送遅延を回避した高速 UDP 通信
   - Block-based XOR 冗長パケット（パラメータ `fec_m` で冗長率制御）
   - 生存率 (Survival Rate) と FEC 復元統計の記録

3. **Adaptive Compression**
   - 帯域と計算負荷に応じた動的圧縮方式選択
   - サポート方式: None, LZ4 (高速), zlib (高圧縮)
   - 実測ベースの最適化: $T_{est} = T_{comp} + (Size_{comp} \times R) / BW$

4. **Mobility-Aware Network Emulation**
   - SUMO シミュレーションによるモビリティトレース生成
   - ノード間距離に基づく動的ネットワーク品質エミュレーション
   - Per-Peer Limitation: HTB + Filter による相手ごとのネットワーク制限
   - 4段階品質ランク（Excellent/Good/Fair/Poor）
   - 詳細: [docs/mobility_aware.md](docs/mobility_aware.md)

#### トポロジー生成

1. **Random Waypoint (RWP)** - 移動ありモデル
   - ノードが移動し続ける標準的な WAFL シナリオ
   - ツール: `utils/generate_rwp_topology.py`

2. **Random Geometric Graph (RGG)** - 静的トポロジー
   - ノード位置固定，純粋なグラフ密度評価用
   - Dense（平均次数 ≥ 10）/ Sparse（平均次数 ≤ 4）
   - ツール: `utils/generate_rgg_topology.py`

3. **SUMO Mobility Trace** - モビリティベース
   - SUMO シミュレーションから抽出したリアルな移動パターン
   - ワンコマンド実行: `mise run sumo`


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
      "id": 0,
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
  "epochs": {"self": 64, "wafl": 4096},
  "contact_pattern": "rwp_n28_a1000_r100_p10_s01.json",
  "wafl_phase": {
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "method": {
    "ssp": {"enabled": true, "ssp_threshold": 0.9},
    "udp": {"enabled": false, "fec_m": 9},
    "compression": {"enabled": false}
  }
}
```

#### 5. 実験実行

```bash
# トポロジー生成（初回のみ）
python utils/generate_rgg_topology.py --nodes 3 --epochs 100 --density dense

# 全ノードへデプロイ
mise run deploy

# 実験開始
mise run start

# 結果収集
mise run collect

# クリーンアップ
mise run stop
```

### ディレクトリ構造

```
WAFL-Testbed/
├── docs/                   # 詳細ドキュメント
│   ├── architecture.md     # システムアーキテクチャ
│   ├── configuration.md    # 設定ガイド
│   ├── setup.md            # セットアップガイド
│   └── usage.md            # 使用方法
├── ctrl/                   # コントロールサーバー
│   ├── main.py             # オーケストレーター
│   ├── execution_config.json  # インフラ設定
│   └── parameters.json     # 実験パラメータ
├── wafl/                   # 実行エージェント
│   ├── src/common/
│   │   ├── main.py         # エージェントメイン
│   │   ├── udp_model_sharing.py  # UDP/FEC 実装
│   │   ├── compression_manager.py  # Adaptive Compression
│   │   └── logger.py       # 構造化ログ
│   └── config/             # エージェント設定（自動生成）
├── utils/                  # ユーティリティ
│   ├── generate_rwp_topology.py     # RWP トポロジー生成
│   ├── generate_rgg_topology.py     # RGG トポロジー生成
│   ├── generate_datasets.py         # データセット生成
│   └── generate_nonIID_filters.py   # Non-IID フィルター
├── data/                   # データ・トポロジー
│   └── contact_pattern/    # 接触パターン JSON
└── results/                # 実験結果
```

### ドキュメント

- [システムアーキテクチャ](docs/architecture.md) - 設計とコンポーネント
- [セットアップガイド](docs/setup.md) - インストールと初期設定
- [設定ガイド](docs/configuration.md) - パラメータ詳細
- [使用方法](docs/usage.md) - 実験実行手順

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

---

## English Version

### Overview

WAFL-Testbed is a research platform for evaluating the **scalability and reliability of synchronization schemes and communication protocols** in large-scale, real-world Wireless Ad-hoc Federated Learning (WAFL) environments using a hybrid physical/container infrastructure.

Unlike traditional simulation-based research, this testbed quantifies **real-world constraints** invisible in simulators: OS/network stack latency, physical resource contention, and exponential synchronization overhead growth.

### Key Features

#### Implemented Protocols & Methods

1. **Semi-Synchronous Protocol (SSP) - Reset Model**
   - Prioritizes learning speed by discarding slow nodes
   - Configurable threshold `ssp_threshold` (e.g., 0.9 = force progress at 90% completion)
   - Detailed wasted computation metrics (`wasted_ms`, `wasted_norm`, `batches_processed`)

2. **UDP + XOR-based FEC (Forward Error Correction)**
   - Avoids TCP retransmission delays with fast UDP transfer
   - Block-based XOR redundancy (redundancy controlled by `fec_m` parameter)
   - Survival rate and FEC recovery statistics tracking

3. **Adaptive Compression**
   - Dynamic compression method selection based on bandwidth and CPU load
   - Supported methods: None, LZ4 (fast), zlib (high compression)
   - Measurement-based optimization: $T_{est} = T_{comp} + (Size_{comp} \times R) / BW$

4. **Mobility-Aware Network Emulation**
   - SUMO simulation-based mobility trace generation
   - Distance-based dynamic network quality emulation
   - Per-Peer Limitation: HTB + Filter for per-destination network constraints
   - 4-tier quality ranks (Excellent/Good/Fair/Poor)
   - Details: [docs/mobility_aware.md](docs/mobility_aware.md)

#### Topology Generation

1. **Random Waypoint (RWP)** - Mobile Model
   - Standard WAFL scenario with continuous node movement
   - Tool: `utils/generate_rwp_topology.py`

2. **Random Geometric Graph (RGG)** - Static Topology
   - Fixed node positions for pure graph density evaluation
   - Dense (avg degree ≥ 10) / Sparse (avg degree ≤ 4)
   - Tool: `utils/generate_rgg_topology.py`

3. **SUMO Mobility Trace** - Mobility-Based
   - Realistic movement patterns extracted from SUMO simulations
   - One-command execution: `mise run sumo`


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
      "id": 0,
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
  "epochs": {"self": 64, "wafl": 4096},
  "contact_pattern": "rwp_n28_a1000_r100_p10_s01.json",
  "wafl_phase": {
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "method": {
    "ssp": {"enabled": true, "ssp_threshold": 0.9},
    "udp": {"enabled": false, "fec_m": 9},
    "compression": {"enabled": false}
  }
}
```

#### 5. Run Experiment

```bash
# Generate topology (first time only)
python utils/generate_rgg_topology.py --nodes 3 --epochs 100 --density dense

# Deploy to all nodes
mise run deploy

# Start experiment
mise run start

# Collect results
mise run collect

# Cleanup
mise run stop
```

### Directory Structure

```
WAFL-Testbed/
├── docs/                   # Detailed documentation
│   ├── architecture.md     # System architecture
│   ├── configuration.md    # Configuration guide
│   ├── setup.md            # Setup guide
│   └── usage.md            # Usage guide
├── ctrl/                   # Control Server
│   ├── main.py             # Orchestrator
│   ├── execution_config.json  # Infrastructure config
│   └── parameters.json     # Experiment parameters
├── wafl/                   # Execution Agents
│   ├── src/common/
│   │   ├── main.py         # Agent main
│   │   ├── udp_model_sharing.py  # UDP/FEC implementation
│   │   ├── compression_manager.py  # Adaptive Compression
│   │   └── logger.py       # Structured logging
│   └── config/             # Agent config (auto-generated)
├── utils/                  # Utilities
│   ├── generate_rwp_topology.py     # RWP topology generation
│   ├── generate_rgg_topology.py     # RGG topology generation
│   ├── generate_datasets.py         # Dataset generation
│   └── generate_nonIID_filters.py   # Non-IID filter generation
├── data/                   # Data & Topology
│   └── contact_pattern/    # Contact pattern JSONs
└── results/                # Experiment results
```

### Documentation

- [System Architecture](docs/architecture.md) - Design and components
- [Setup Guide](docs/setup.md) - Installation and initial setup
- [Configuration Guide](docs/configuration.md) - Parameter details
- [Usage Guide](docs/usage.md) - Experiment execution

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
