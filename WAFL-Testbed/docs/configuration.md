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
      "id": 0,
      "physical_ip": "192.168.11.100",
      "container_port_ctrl": 10001,
      "host_port_ctrl": 10001,
      "host_port_p2p": 10002,
      "cpu_limit": "1.0"
    },
    {
      "id": 1,
      "physical_ip": "192.168.11.101",
      "container_port_ctrl": 10001,
      "host_port_ctrl": 11001,
      "host_port_p2p": 11002,
      "cpu_limit": "0.5"
    },
    {
      "id": 2,
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
| `id`                  | integer | ✓    | エージェントの一意 ID (0〜N)         | `0`, `1`, `2`      |
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
    {"id": 0, "physical_ip": "192.168.11.100", "host_port_ctrl": 10001, "host_port_p2p": 10002},
    {"id": 1, "physical_ip": "192.168.11.100", "host_port_ctrl": 11001, "host_port_p2p": 11002},
    {"id": 2, "physical_ip": "192.168.11.100", "host_port_ctrl": 12001, "host_port_p2p": 12002},
    {"id": 3, "physical_ip": "192.168.11.100", "host_port_ctrl": 13001, "host_port_p2p": 13002}
  ]
}
```

### 2. `ctrl/parameters.json` - 実験パラメータ

実験設定，ハイパーパラメータ，ネットワーク条件，アルゴリズムを定義する．

#### 設定例

```json
{
  "epochs": {
    "self": 64,
    "wafl": 4096
  },
  "contact_pattern": "rwp_n28_a1000_r100_p10_s01.json",
  "wafl_phase": {
    "aggregation_strategy": "FedAvg",
    "batch_size": 32,
    "learning_rate": 0.001,
    "coefficiency": 1.0
  },
  "network_condition": {
    "enabled": true,
    "delay": "50ms",
    "loss": "3%",
    "rate": "10mbit"
  },
  "method": {
    "ssp": {
      "enabled": true,
      "staleness": 5,
      "ssp_threshold": 0.9
    },
    "udp": {
      "enabled": true,
      "fec_m": 9
    },
    "compression": {
      "enabled": true,
      "initial_method": "zlib"
    }
  }
}
```

#### パラメータ詳細

**`epochs`** (object): エポック数設定

| パラメータ | 説明                                  | 推奨値     |
| ---------- | ------------------------------------- | ---------- |
| `self`     | SELF フェーズのエポック数（独立学習） | 64〜100    |
| `wafl`     | WAFL フェーズのエポック数（連合学習） | 1000〜5000 |

**`contact_pattern`** (string): 接触パターンファイル名（`data/contact_pattern/` 内）

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
| `delay`    | ネットワーク遅延（レイテンシ） | `"50ms"`, `"100ms"`                | `delay 50ms`  |
| `loss`     | パケットロス率                 | `"0%"`, `"3%"`, `"10%"`            | `loss 3%`     |
| `rate`     | 帯域制限                       | `"100mbit"`, `"10mbit"`, `"1mbit"` | `rate 10mbit` |

**注意**: `enabled` を `false` に設定すると、`delay`、`loss`、`rate` の設定に関わらず、ネットワーク条件のエミュレーションがスキップされます。デフォルトは `true` です。

### 3. `method` セクション - 研究手法の設定

#### 3.1 SSP (Semi-Synchronous Protocol)

**目的**: 遅延ノードを切り捨てて学習速度を優先

```json
{
  "ssp": {
    "enabled": true,
    "staleness": 5,
    "ssp_threshold": 0.9
  }
}
```

| パラメータ      | 説明                     | 範囲           | 推奨値    |
| --------------- | ------------------------ | -------------- | --------- |
| `enabled`       | SSP 有効化               | `true`/`false` | -         |
| `staleness`     | 許容する最大エポック差   | 0〜999         | 5〜10     |
| `ssp_threshold` | 強制進行の閾値（完了率） | 0.0〜1.0       | 0.8〜0.95 |

**動作**:
- 完了ノード数が `ノード総数 × ssp_threshold` に達したら，未完了ノードに `FORCE_NEXT` を送信
- 未完了ノードは現在の計算を破棄して次エポックへ進む
- 破棄された計算量（`wasted_ms`, `wasted_norm`）がログに記録される

**実験例**:
| ノード数 | 閾値 | 強制進行タイミング |
| -------- | ---- | ------------------ |
| 10       | 0.9  | 9 ノード完了時     |
| 28       | 0.8  | 23 ノード完了時    |
| 100      | 0.95 | 95 ノード完了時    |

#### 3.2 UDP + FEC

**目的**: TCP 再送遅延を回避した高速モデル転送

```json
{
  "udp": {
    "enabled": true,
    "fec_m": 9
  }
}
```

| パラメータ | 説明                                                       | 範囲           | 推奨値 |
| ---------- | ---------------------------------------------------------- | -------------- | ------ |
| `enabled`  | UDP モード有効化                                           | `true`/`false` | -      |
| `fec_m`    | FEC パラメータ (M 個のデータチャンクに 1 個の冗長パケット) | 4〜19          | 9      |

**FEC 冗長率**:
$$\text{冗長率} = \frac{1}{M+1}$$

| `fec_m` | 冗長率 | 用途                       |
| ------- | ------ | -------------------------- |
| 4       | 20%    | 高損失環境 (Loss 5%〜10%)  |
| 9       | 10%    | 中損失環境 (Loss 1%〜5%)   |
| 19      | 5%     | 低損失環境 (Loss 0.1%〜1%) |

**復元理論**:
- 1ブロック内で最大 1 パケットロスまで復元可能
- 2 パケット以上ロス時は復元不可 → モデル破棄
- モデル生存率: $(1 - P_{block\_fail})^B$（B: 総ブロック数）

#### 3.3 Adaptive Compression

**目的**: 帯域と計算負荷に応じた動的圧縮方式選択

```json
{
  "compression": {
    "enabled": true,
    "initial_method": "zlib"
  }
}
```

| パラメータ       | 説明         | 選択肢                      |
| ---------------- | ------------ | --------------------------- |
| `enabled`        | 圧縮有効化   | `true`/`false`              |
| `initial_method` | 初期圧縮方式 | `"none"`, `"lz4"`, `"zlib"` |

**圧縮方式の特性**:

| 方式   | 圧縮率          | 速度    | 用途                |
| ------ | --------------- | ------- | ------------------- |
| `none` | 1.0（圧縮なし） | -       | 高帯域・低 CPU 環境 |
| `lz4`  | 0.5             | 500MB/s | バランス型          |
| `zlib` | 0.3             | 50MB/s  | 低帯域・高 CPU 環境 |

**動的選択ロジック**:

推定転送時間を最小化:
$$T_{est} = T_{comp} + \frac{Size_{comp} \times R}{BW}$$

where:
- $T_{comp}$: 圧縮時間
- $Size_{comp}$: 圧縮後サイズ
- $R$: FEC 冗長率 = $1 + 1/(M+1)$
- $BW$: 推定帯域（EMA で更新）

### 実験シナリオ例

#### シナリオ 1: Baseline（厳密同期 + TCP）

```json
{
  "method": {
    "ssp": {"enabled": false},
    "udp": {"enabled": false},
    "compression": {"enabled": false}
  },
  "network_condition": {"delay": "50ms", "loss": "0%", "rate": "100mbit"}
}
```

#### シナリオ 2: SSP 最適化

```json
{
  "method": {
    "ssp": {"enabled": true, "ssp_threshold": 0.9},
    "udp": {"enabled": false},
    "compression": {"enabled": false}
  }
}
```

#### シナリオ 3: UDP + FEC（劣化環境）

```json
{
  "method": {
    "ssp": {"enabled": false},
    "udp": {"enabled": true, "fec_m": 4},
    "compression": {"enabled": false}
  },
  "network_condition": {"delay": "50ms", "loss": "5%", "rate": "10mbit"}
}
```

#### シナリオ 4: 統合最適化

```json
{
  "method": {
    "ssp": {"enabled": true, "ssp_threshold": 0.8},
    "udp": {"enabled": true, "fec_m": 9},
    "compression": {"enabled": true, "initial_method": "lz4"}
  },
  "network_condition": {"delay": "50ms", "loss": "3%", "rate": "10mbit"}
}
```

---

## English Version

### 1. `ctrl/execution_config.json` - Infrastructure

Defines physical infrastructure, node topology, and resource limits.

#### Parameters

**`nodes`**: Array of agent configuration objects

| Parameter             | Type    | Required | Description                   | Example            |
| --------------------- | ------- | -------- | ----------------------------- | ------------------ |
| `id`                  | integer | ✓        | Unique agent ID (0〜N)        | `0`, `1`           |
| `physical_ip`         | string  | ✓        | Physical server IP address    | `"192.168.11.100"` |
| `container_port_ctrl` | integer | ✓        | Control port inside container | `10001`            |
| `host_port_ctrl`      | integer | ✓        | Control port on host (mapped) | `10001`            |
| `host_port_p2p`       | integer | ✓        | P2P port on host              | `10002`            |
| `cpu_limit`           | string  | -        | CPU limit (cores)             | `"1.0"`, `"0.5"`   |

### 2. `ctrl/parameters.json` - Experiment

Defines experiment settings, hyperparameters, network conditions, and algorithms.

#### Method Configuration

**SSP (Semi-Synchronous Protocol)**:
- `ssp_threshold`: Completion rate threshold (0.0〜1.0)
- `staleness`: Maximum allowed epoch difference

**UDP + FEC**:
- `fec_m`: FEC parameter (M data chunks per parity packet)
- Redundancy rate: $1/(M+1)$

**Adaptive Compression**:
- `initial_method`: Initial compression method (`"none"`, `"lz4"`, `"zlib"`)
- Dynamic selection: minimize $T_{est} = T_{comp} + \frac{Size_{comp} \times R}{BW}$

### Experiment Scenarios

See Japanese version for detailed scenarios.
