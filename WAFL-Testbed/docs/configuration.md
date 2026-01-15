# 設定ガイド / Configuration Guide

**日本語** | [English](#english-version)

---

## 日本語版

WAFL-Testbed は 2 つの主要 JSON 設定ファイルで構成される．

### 1. `ctrl/execution_config.json` - インフラ設定

物理インフラ，ノードトポロジー，リソース制限を定義する．

#### 設定例

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
    },
    {
      "name": 1,
      "physical_ip": "192.168.11.101",
      "container_port_ctrl": 10001,
      "host_port_ctrl": 11001,
      "host_port_p2p": 11002,
      "cpu_limit": "0.5"
    },
    {
      "name": 2,
      "physical_ip": "192.168.11.102",
      "container_port_ctrl": 10001,
      "host_port_ctrl": 12001,
      "host_port_p2p": 12002
    }
  ],
  "deployment_location": "/home/denjo",
  "user": "denjo"
}
```

#### パラメータ詳細

**`nodes`** (配列): 各エージェントの設定オブジェクトリスト

| パラメータ            | 型      | 必須 | 説明                                 | 例                 |
| --------------------- | ------- | ---- | ------------------------------------ | ------------------ |
| `name`                | integer | ✓    | エージェントの一意 ID (0〜N)         | `0`, `1`, `2`      |
| `physical_ip`         | string  | ✓    | 物理サーバーの IP アドレス           | `"192.168.11.100"` |
| `container_port_ctrl` | integer | ✓    | コンテナ内の制御ポート               | `10001`            |
| `host_port_ctrl`      | integer | ✓    | ホスト側の制御ポート（マッピング先） | `10001`, `11001`   |
| `host_port_p2p`       | integer | ✓    | ホスト側の P2P ポート                | `10002`, `11002`   |
| `cpu_limit`           | string  | -    | CPU 制限 (コア数)                    | `"1.0"`, `"0.5"`   |

**`deployment_location`** (string): リモートサーバー上の配置先ディレクトリ（絶対パス）

**`user`** (string): 実行サーバーへの SSH ユーザー名

#### 複数コンテナ配置例（同一物理サーバー上）

```json
{
  "nodes": [
    {"name": 0, "physical_ip": "192.168.11.100", "host_port_ctrl": 10001, "host_port_p2p": 10002},
    {"name": 1, "physical_ip": "192.168.11.100", "host_port_ctrl": 11001, "host_port_p2p": 11002},
    {"name": 2, "physical_ip": "192.168.11.100", "host_port_ctrl": 12001, "host_port_p2p": 12002},
    {"name": 3, "physical_ip": "192.168.11.100", "host_port_ctrl": 13001, "host_port_p2p": 13002}
  ]
}
```

---

### 2. `ctrl/parameters.json` - 実験パラメータ

実験設定，ハイパーパラメータ，ネットワーク条件，通信モードを定義する．

#### 設定例

```json
{
  "experiment_name": "Experiment 0: excellent (dynamic)",
  "epochs": {
    "self": 8,
    "wafl": 64
  },
  "contact_pattern": "contact_pattern/rwp_n50_a1000_r100_p10_s01.json",
  "wafl_phase": {
    "aggregation_strategy": "FedAvg",
    "batch_size": 32,
    "learning_rate": 0.001,
    "coefficiency": 1.0
  },
  "network_condition": {
    "enabled": true,
    "rate": "10mbit",
    "delay": "5ms",
    "loss": "2%"
  },
  "mobility_aware": {
    "enabled": false
  },
  "method": "dynamic",
  "ssp": {
    "enabled": true,
    "ssp_threshold": 0.8
  }
}
```

#### パラメータ詳細

**`experiment_name`** (string): 実験の識別名（結果ディレクトリ名に使用）

**`epochs`** (object): エポック数設定

| パラメータ | 説明                                  | 推奨値   |
| ---------- | ------------------------------------- | -------- |
| `self`     | SELF フェーズのエポック数（独立学習） | 8〜64    |
| `wafl`     | WAFL フェーズのエポック数（連合学習） | 64〜4096 |

**`contact_pattern`** (string): 接触パターンファイル名（`data/` からの相対パス）

生成方法:
```bash
# RWP (移動あり)
python utils/generate_rwp_topology.py --nodes 28 --times 5000

# RGG Dense (静的・高密度)
python utils/generate_rgg_topology.py --nodes 28 --epochs 5000 --density dense

# RGG Sparse (静的・低密度)
python utils/generate_rgg_topology.py --nodes 28 --epochs 5000 --density sparse
```

**`wafl_phase`** (object): 学習アルゴリズムのハイパーパラメータ

| パラメータ             | 説明                 | デフォルト |
| ---------------------- | -------------------- | ---------- |
| `aggregation_strategy` | モデル集約戦略       | `"FedAvg"` |
| `batch_size`           | バッチサイズ         | `32`       |
| `learning_rate`        | 学習率               | `0.001`    |
| `coefficiency`         | モデル差分の混合係数 | `1.0`      |

**`network_condition`** (object): ネットワークエミュレーション設定（グローバル）

| パラメータ | 説明                           | 例                                 | tc コマンド   |
| ---------- | ------------------------------ | ---------------------------------- | ------------- |
| `enabled`  | ネットワーク条件の有効化       | `true`, `false`                    | -             |
| `delay`    | ネットワーク遅延（レイテンシ） | `"5ms"`, `"50ms"`, `"100ms"`       | `delay 50ms`  |
| `loss`     | パケットロス率                 | `"0%"`, `"2%"`, `"5%"`, `"10%"`    | `loss 3%`     |
| `rate`     | 帯域制限                       | `"100mbit"`, `"10mbit"`, `"1mbit"` | `rate 10mbit` |

> **Note**: `enabled` を `false` に設定すると、`delay`、`loss`、`rate` の設定に関わらず、ネットワーク条件のエミュレーションがスキップされる．

---

### 3. 通信モード (`method`) の設定

WAFL-Testbed は 3 つの通信モードを提供する．`method` パラメータで選択する．

#### 3.1 TCP モード

```json
{
  "method": "tcp"
}
```

| 項目           | 内容                                       |
| -------------- | ------------------------------------------ |
| **プロトコル** | 標準 TCP 接続                              |
| **信頼性**     | TCP による完全保証                         |
| **圧縮**       | なし                                       |
| **FEC**        | なし                                       |
| **適用環境**   | 安定したネットワーク環境，ベースライン比較 |

**特徴**:
- シンプルで信頼性の高い通信
- パケットロス時は TCP 再送により復元
- 高損失環境では再送遅延が累積しやすい

#### 3.2 Dynamic モード

```json
{
  "method": "dynamic"
}
```

| 項目           | 内容                                 |
| -------------- | ------------------------------------ |
| **プロトコル** | UDP + zfec FEC                       |
| **信頼性**     | FEC + Proactive NACK                 |
| **圧縮**       | Adaptive（zlib/LZ4/none を動的選択） |
| **FEC**        | ネットワーク品質に応じた適応的冗長度 |
| **適用環境**   | 不安定なネットワーク環境             |

**特徴**:
- ネットワーク状態を実測し，パラメータを自動調整
- 高損失環境では FEC 冗長度を増加
- 低帯域環境では圧縮を強化（zlib）
- 高帯域環境では圧縮を軽減（LZ4）または無効化

**内部パラメータ（自動調整）**:
- FEC 冗長度: 損失率に応じて 6.25%〜25%
- 圧縮方式: 帯域に応じて zlib/LZ4/none
- ペーシング間隔: RTT に応じて動的調整

#### 3.3 Fast モード

```json
{
  "method": "fast"
}
```

| 項目           | 内容                       |
| -------------- | -------------------------- |
| **プロトコル** | UDP + zfec FEC（高速設定） |
| **信頼性**     | 最小限の FEC               |
| **圧縮**       | LZ4（条件付きスキップ）    |
| **FEC**        | 低冗長度（6.25%固定）      |
| **適用環境**   | 高帯域・低損失環境         |

**特徴**:
- 最小限のオーバーヘッドで高速転送を実現
- 高帯域環境では圧縮をスキップ（帯域 > 50Mbps）
- FEC 冗長度は固定で最小限
- 損失率が高い環境では性能が劣化する可能性あり

#### 3 モードの比較

| 比較項目           | TCP    | Dynamic        | Fast     |
| ------------------ | ------ | -------------- | -------- |
| **転送速度**       | 中     | 中〜高         | 高       |
| **信頼性**         | 最高   | 高             | 中       |
| **適応性**         | -      | 高             | 低       |
| **オーバーヘッド** | 低     | 中             | 最低     |
| **推奨損失率**     | 0%〜3% | 1%〜10%        | 0%〜2%   |
| **推奨帯域**       | 任意   | 1Mbps〜100Mbps | 10Mbps〜 |

---

### 4. SSP (Semi-Synchronous Protocol) の設定

**目的**: 遅延ノードを切り捨てて学習速度を優先（1エポック以上の遅れは発生しない）

```json
{
  "ssp": {
    "enabled": true,
    "ssp_threshold": 0.8
  }
}
```

| パラメータ      | 説明                     | 範囲           | 推奨値    |
| --------------- | ------------------------ | -------------- | --------- |
| `enabled`       | SSP 有効化               | `true`/`false` | -         |
| `ssp_threshold` | 強制進行の閾値（完了率） | 0.0〜1.0       | 0.8〜0.95 |

**動作**:
- 完了ノード数が `ノード総数 × ssp_threshold` に達したら，未完了ノードに `FORCE_NEXT` を送信
- 未完了ノードは現在の計算を破棄して同じエポックにスキップ
- これにより、1エポック以上遅れるノードは存在しない
- 破棄された計算量（`wasted_ms`, `wasted_norm`）がログに記録される

**実験例**:
| ノード数 | 閾値 | 強制進行タイミング |
| -------- | ---- | ------------------ |
| 10       | 0.8  | 8 ノード完了時     |
| 28       | 0.8  | 23 ノード完了時    |
| 50       | 0.9  | 45 ノード完了時    |

> **Note**: `ssp.enabled` が `false` の場合，BSP（Bulk Synchronous Parallel）として動作し，全ノードの完了を待機する．

---

### 5. Mobility-Aware の設定

動的ネットワーク条件（ノード間距離に応じた通信品質変化）を有効化する．

```json
{
  "mobility_aware": {
    "enabled": true,
    "contact_pattern_file": "contact_pattern_mobility.json",
    "network_conditions_file": "network_conditions_mobility.json",
    "path_loss_model_file": "sumo/path_loss_model.json"
  }
}
```

詳細: [docs/mobility_aware.md](mobility_aware.md)

---

### 6. 実験シナリオ例

#### シナリオ 1: Baseline（TCP + BSP）

安定したネットワーク環境での厳密同期ベースライン．

```json
{
  "experiment_name": "Baseline TCP",
  "method": "tcp",
  "ssp": {"enabled": false},
  "network_condition": {"enabled": true, "delay": "5ms", "loss": "0%", "rate": "100mbit"}
}
```

#### シナリオ 2: TCP + SSP

TCP 通信を維持しつつ，遅延ノードを切り捨て．

```json
{
  "experiment_name": "TCP with SSP",
  "method": "tcp",
  "ssp": {"enabled": true, "ssp_threshold": 0.8},
  "network_condition": {"enabled": true, "delay": "50ms", "loss": "3%", "rate": "10mbit"}
}
```

#### シナリオ 3: Dynamic（劣化環境）

不安定なネットワーク環境での適応的通信．

```json
{
  "experiment_name": "Dynamic Mode - Fair",
  "method": "dynamic",
  "ssp": {"enabled": true, "ssp_threshold": 0.8},
  "network_condition": {"enabled": true, "delay": "50ms", "loss": "5%", "rate": "5mbit"}
}
```

#### シナリオ 4: Fast（高帯域環境）

高帯域環境での高速通信．

```json
{
  "experiment_name": "Fast Mode - Excellent",
  "method": "fast",
  "ssp": {"enabled": true, "ssp_threshold": 0.8},
  "network_condition": {"enabled": true, "delay": "5ms", "loss": "1%", "rate": "100mbit"}
}
```

#### シナリオ 5: Mobility-Aware

SUMO モビリティトレースに基づく動的ネットワーク条件．

```json
{
  "experiment_name": "Mobility-Aware Experiment",
  "contact_pattern": "sumo/contact_pattern_mobility.json",
  "method": "dynamic",
  "ssp": {"enabled": true, "ssp_threshold": 0.8},
  "mobility_aware": {
    "enabled": true,
    "contact_pattern_file": "contact_pattern_mobility.json",
    "network_conditions_file": "network_conditions_mobility.json",
    "path_loss_model_file": "sumo/path_loss_model.json"
  }
}
```

---

### 7. 複数実験の自動実行

`ctrl/parameters/` ディレクトリにパラメータファイルを配置し，`mise experiments` で順次実行できる．

```
ctrl/parameters/
├── exp0_1-excellent-1-tcp.json
├── exp0_1-excellent-3-dynamic.json
├── exp0_1-excellent-4-fast.json
├── exp0_2-good-1-tcp.json
├── exp0_2-good-3-dynamic.json
...
```

詳細: [docs/usage.md](usage.md)

---

## English Version

### 1. `ctrl/execution_config.json` - Infrastructure

Defines physical infrastructure, node topology, and resource limits.

#### Parameters

**`nodes`**: Array of agent configuration objects

| Parameter             | Type    | Required | Description                   | Example            |
| --------------------- | ------- | -------- | ----------------------------- | ------------------ |
| `name`                | integer | ✓        | Unique agent ID (0〜N)        | `0`, `1`           |
| `physical_ip`         | string  | ✓        | Physical server IP address    | `"192.168.11.100"` |
| `container_port_ctrl` | integer | ✓        | Control port inside container | `10001`            |
| `host_port_ctrl`      | integer | ✓        | Control port on host (mapped) | `10001`            |
| `host_port_p2p`       | integer | ✓        | P2P port on host              | `10002`            |
| `cpu_limit`           | string  | -        | CPU limit (cores)             | `"1.0"`, `"0.5"`   |

---

### 2. `ctrl/parameters.json` - Experiment

Defines experiment settings, hyperparameters, network conditions, and algorithms.

---

### 3. Communication Mode Configuration

#### TCP Mode
```json
{"method": "tcp"}
```
Standard TCP connections, reliable but slower under packet loss.

#### Dynamic Mode
```json
{"method": "dynamic"}
```
Adaptive UDP+FEC with automatic parameter tuning based on network conditions.

#### Fast Mode
```json
{"method": "fast"}
```
High-speed UDP+FEC optimized for high-bandwidth, low-loss environments.

---

### 4. SSP (Semi-Synchronous Protocol) Configuration

- `ssp_threshold`: Completion rate threshold (0.0〜1.0)
- When threshold is reached, slow nodes are force-skipped to ensure no node is more than 1 epoch behind

---

### 5. Experiment Scenarios

See Japanese version for detailed scenarios.
