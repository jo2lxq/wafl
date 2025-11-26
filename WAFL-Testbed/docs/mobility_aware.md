# モビリティ対応ネットワークエミュレーション / Mobility-Aware Network Emulation

**日本語** | [English](#english-version)

---

## 日本語版

### 概要

従来の静的ネットワーク条件（一定の帯域・遅延・ロス率）に対し，**ノード間距離に応じて通信品質がリアルタイムに変化**する環境を実現する．これにより，実社会の移動体通信（VANET 等）におけるフェージングや切断プロセスを模擬できる．

#### 主な機能

- **SUMO 統合**: SUMO シミュレーションからモビリティトレースを抽出
- **距離ベース品質計算**: ノード間距離から通信品質（帯域・遅延・ロス率）を自動計算
- **Per-Peer Limitation**: HTB + Filter を使った相手ごとのネットワーク制限
- **4 段階ランク制**: Excellent/Good/Fair/Poor の品質ランク
- **設定可能な距離減衰モデル**: JSON ファイルでカスタマイズ可能
- **mise タスク統合**: `mise run sumo` で前計算を手軽に実行

### 使用方法

#### 1. SUMO のインストール

```bash
# Ubuntu/Debian
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

# 環境変数設定
export SUMO_HOME="/usr/share/sumo"
```

公式ドキュメント: https://sumo.dlr.de/docs/Installing/index.html

#### 2. 実行設定から SUMO ルートを生成

`execution_config.json` からノード情報を読み込み，SUMO ルートファイルを動的生成する．これにより，ノード数の変更が自動的に SUMO シナリオに反映される．

```bash
# execution_config.json からルート生成（ネットワークファイルも自動生成）
mise run generate-routes
```

生成されるファイル:
- `data/sumo/routes.rou.xml`: 車両ルート定義（ノード数と ID が execution_config.json と一致）
- `data/sumo/network.net.xml`: 円形道路ネットワーク（自動生成）
- `data/sumo/routes.sumocfg`: SUMO 設定ファイル

**メリット:**
- **ID の一致**: execution_config.json の `name: 15` と SUMO の `vehicle id="15"` が一致
- **スケーラビリティ**: ノード数を変更すると自動的に SUMO シナリオも更新
- **ミス防止**: ノード数の不整合を回避

#### 3. SUMO シナリオの作成（手動の場合）

手動で SUMO シナリオを作成する場合，モビリティシナリオ（`.sumocfg`，`.rou.xml`，`.net.xml`）を `data/sumo/` に配置する．

例: 簡単な交差点シナリオ

```bash
# data/sumo/ ディレクトリ構造例
data/sumo/
├── scenario.sumocfg      # SUMO 設定ファイル
├── network.net.xml       # 道路ネットワーク
└── routes.rou.xml        # 車両ルート
```

#### 3. モビリティトレースの生成

```bash
# 環境変数でパス指定（オプション）
export SUMO_CONFIG=data/sumo/scenario.sumocfg
export TRACE_OUTPUT=data/sumo/mobility_trace.csv

# SUMO シミュレーション実行 → sumo/mobility_trace.csv 生成
mise run generate-sumo-trace
```

出力: `data/sumo/mobility_trace.csv`

```csv
epoch,node_id,x,y
0,0,100.50,200.30
0,1,150.20,180.40
1,0,102.30,201.50
1,1,148.80,182.10
...
```

#### 4. コンタクトパターンとネットワーク条件の計算

```bash
mise run sumo
```

生成ファイル:
- `data/contact_pattern_mobility.json`: エポックごとの通信ペアリスト
- `data/network_conditions_mobility.json`: エポックごとの tc 設定

#### 5. 距離減衰モデルのカスタマイズ

`data/sumo/path_loss_model.json` を作成：

```json
{
  "radio_range": 150,
  "ranks": [
    {
      "name": "excellent",
      "distance_min": 0,
      "distance_max": 30,
      "rate": "100mbit",
      "delay": "5ms",
      "loss": "0%"
    },
    {
      "name": "good",
      "distance_min": 30,
      "distance_max": 80,
      "rate": "20mbit",
      "delay": "20ms",
      "loss": "0.5%"
    }
  ]
}
```

- `radio_range`: 無線通信可能な最大距離（メートル）
- `ranks`: 距離帯ごとのネットワーク品質定義

#### 6. 実験での利用

`ctrl/parameters.json` で mobility_aware モードを有効化：

```json
{
  "contact_pattern": "contact_pattern_mobility.json",
  "mobility_aware": {
    "enabled": true,
    "contact_pattern_file": "contact_pattern_mobility.json",
    "network_conditions_file": "network_conditions_mobility.json",
    "path_loss_model_file": "sumo/path_loss_model.json"
  }
}
```

通常通り実験を実行：

```bash
mise run deploy
mise run start
mise run collect
mise run stop
```

### Per-Peer Limitation の技術詳細

HTB（Hierarchical Token Bucket）+ Filter を使った実装により，**通信相手ごとに異なるネットワーク制限**を実現する．

#### 実装方式

1. **HTB ルート作成**: 階層的キューイングの基盤
2. **ランククラス作成**: 4 段階（excellent/good/fair/poor）のクラス
3. **NetEm 適用**: 各クラスに遅延・ロス率を設定
4. **IP フィルタリング**: 宛先 IP アドレスで該当クラスへルーティング

#### tc コマンド例

```bash
# 1. ルート HTB 作成
tc qdisc add dev vethXXXX root handle 1: htb

# 2. ランクごとのクラス作成
tc class add dev vethXXXX parent 1: classid 1:1 htb rate 100mbit  # excellent
tc class add dev vethXXXX parent 1: classid 1:2 htb rate 20mbit   # good

# 3. NetEm をクラスに適用
tc qdisc add dev vethXXXX parent 1:1 handle 10: netem delay 5ms loss 0%
tc qdisc add dev vethXXXX parent 1:2 handle 20: netem delay 20ms loss 0.5%

# 4. 相手 IP を該当クラスへフィルタリング
tc filter add dev vethXXXX protocol ip parent 1:0 prio 1 u32 match ip dst 172.18.0.2 flowid 1:1
tc filter add dev vethXXXX protocol ip parent 1:0 prio 1 u32 match ip dst 172.18.0.3 flowid 1:2
```

#### 効果

- 通信相手 A とは 100Mbps（excellent）で通信
- 同時に通信相手 B とは 5Mbps（fair）で通信
- 提案手法（Adaptive Compression, UDP/FEC）の効果を適切に評価可能

### トラブルシューティング

#### SUMO が見つからない

```bash
# SUMO_HOME 環境変数を設定
export SUMO_HOME="/usr/share/sumo"
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
```

#### tc コマンドの権限エラー

`apply_dynamic_tc.py` は `sudo` を使用する．パスワードなし sudo 設定が必要な場合：

```bash
# /etc/sudoers に追加（visudo で編集）
your_username ALL=(ALL) NOPASSWD: /sbin/tc
```

#### 動的 tc 設定が適用されない

1. `mobility_aware.enabled` が `true` か確認
2. ログファイルで "Applying dynamic network conditions" メッセージを確認
3. ホスト上で手動実行テスト:
   ```bash
   cd /path/to/wafl
   python3 utils/apply_dynamic_tc.py \
     --container wafl-node-0 \
     --epoch 0 \
     --node-id 0 \
     --conditions data/network_conditions_mobility.json \
     --pathloss data/sumo/path_loss_model.json
   ```

---

## English Version

### Overview

Instead of static network conditions (constant bandwidth, delay, and packet loss), this feature enables **real-time changes in communication quality based on inter-node distances**. This simulates fading and disconnection processes in real-world mobile communications (VANET, etc.).

#### Key Features

- **SUMO Integration**: Extract mobility traces from SUMO simulations
- **Distance-Based Quality Calculation**: Automatically calculate communication quality (bandwidth, delay, packet loss) from inter-node distances
- **Per-Peer Limitation**: Per-destination network constraints using HTB + Filter
- **4-Tier Ranking System**: Excellent/Good/Fair/Poor quality ranks
- **Configurable Path Loss Model**: Customizable via JSON configuration file
- **mise Task Integration**: Easy preprocessing execution with `mise run sumo`

### Usage

#### 1. Install SUMO

```bash
# Ubuntu/Debian
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

# Set environment variable
export SUMO_HOME="/usr/share/sumo"
```

Official documentation: https://sumo.dlr.de/docs/Installing/index.html

#### 2. Generate SUMO Routes from Configuration

Dynamically generate SUMO route files from `execution_config.json`. This ensures that changes in node count are automatically reflected in the SUMO scenario.

```bash
# Generate routes from execution_config.json (network file auto-generated)
mise run generate-routes
```

Generated files:
- `data/sumo/routes.rou.xml`: Vehicle route definitions (node count and IDs match execution_config.json)
- `data/sumo/network.net.xml`: Circular road network (auto-generated)
- `data/sumo/routes.sumocfg`: SUMO configuration file

**Benefits:**
- **ID Consistency**: `name: 15` in execution_config.json matches `vehicle id="15"` in SUMO
- **Scalability**: Changing node count automatically updates SUMO scenario
- **Error Prevention**: Avoids node count mismatches

#### 3. Create SUMO Scenario (Manual Method)

For manual SUMO scenario creation, place mobility scenarios (`.sumocfg`, `.rou.xml`, `.net.xml`) in `data/sumo/`.

Example: Simple intersection scenario

```bash
# Example data/sumo/ directory structure
data/sumo/
├── scenario.sumocfg      # SUMO configuration file
├── network.net.xml       # Road network
└── routes.rou.xml        # Vehicle routes
```

#### 3. Generate Mobility Trace

```bash
# Run SUMO simulation → generate sumo/mobility_trace.csv
mise run generate-sumo-trace
```

Output: `data/sumo/mobility_trace.csv`

```csv
epoch,node_id,x,y
0,0,100.50,200.30
0,1,150.20,180.40
1,0,102.30,201.50
1,1,148.80,182.10
...
```

#### 4. Calculate Contact Pattern and Network Conditions

```bash
mise run sumo
```

Generated files:
- `data/contact_pattern_mobility.json`: Communication pair list per epoch
- `data/network_conditions_mobility.json`: tc settings per epoch

#### 5. Customize Path Loss Model

Create `data/sumo/path_loss_model.json`:

```json
{
  "radio_range": 150,
  "ranks": [
    {
      "name": "excellent",
      "distance_min": 0,
      "distance_max": 30,
      "rate": "100mbit",
      "delay": "5ms",
      "loss": "0%"
    },
    {
      "name": "good",
      "distance_min": 30,
      "distance_max": 80,
      "rate": "20mbit",
      "delay": "20ms",
      "loss": "0.5%"
    }
  ]
}
```

- `radio_range`: Maximum wireless communication range (meters)
- `ranks`: Network quality definitions per distance range

#### 6. Use in Experiments

Enable mobility_aware mode in `ctrl/parameters.json`:

```json
{
  "contact_pattern": "contact_pattern_mobility.json",
  "mobility_aware": {
    "enabled": true,
    "contact_pattern_file": "contact_pattern_mobility.json",
    "network_conditions_file": "network_conditions_mobility.json",
    "path_loss_model_file": "sumo/path_loss_model.json"
  }
}
```

Run experiments as usual:

```bash
mise run deploy
mise run start
mise run collect
mise run stop
```

### Technical Details: Per-Peer Limitation

The implementation using HTB (Hierarchical Token Bucket) + Filter achieves **different network constraints per communication peer**.

#### Implementation Approach

1. **Create HTB Root**: Foundation for hierarchical queuing
2. **Create Rank Classes**: 4-tier classes (excellent/good/fair/poor)
3. **Apply NetEm**: Set delay and loss rate for each class
4. **IP Filtering**: Route to appropriate class based on destination IP address

#### Example tc Commands

```bash
# 1. Create root HTB
tc qdisc add dev vethXXXX root handle 1: htb

# 2. Create classes per rank
tc class add dev vethXXXX parent 1: classid 1:1 htb rate 100mbit  # excellent
tc class add dev vethXXXX parent 1: classid 1:2 htb rate 20mbit   # good

# 3. Apply NetEm to classes
tc qdisc add dev vethXXXX parent 1:1 handle 10: netem delay 5ms loss 0%
tc qdisc add dev vethXXXX parent 1:2 handle 20: netem delay 20ms loss 0.5%

# 4. Filter peer IPs to appropriate classes
tc filter add dev vethXXXX protocol ip parent 1:0 prio 1 u32 match ip dst 172.18.0.2 flowid 1:1
tc filter add dev vethXXXX protocol ip parent 1:0 prio 1 u32 match ip dst 172.18.0.3 flowid 1:2
```

#### Benefits

- Communicate with peer A at 100Mbps (excellent)
- Simultaneously communicate with peer B at 5Mbps (fair)
- Properly evaluate the effectiveness of proposed methods (Adaptive Compression, UDP/FEC)

### Troubleshooting

#### SUMO Not Found

```bash
# Set SUMO_HOME environment variable
export SUMO_HOME="/usr/share/sumo"
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
```

#### tc Command Permission Error

`apply_dynamic_tc.py` uses `sudo`. For passwordless sudo if needed:

```bash
# Add to /etc/sudoers (edit with visudo)
your_username ALL=(ALL) NOPASSWD: /sbin/tc
```

#### Dynamic tc Configuration Not Applied

1. Confirm `mobility_aware.enabled` is `true`
2. Check for "Applying dynamic network conditions" messages in log files
3. Test manual execution on host:
   ```bash
   cd /path/to/wafl
   python3 utils/apply_dynamic_tc.py \
     --container wafl-node-0 \
     --epoch 0 \
     --node-id 0 \
     --conditions data/network_conditions_mobility.json \
     --pathloss data/sumo/path_loss_model.json
   ```

### References

- **SUMO Official**: https://sumo.dlr.de/
- **TraCI (Python API)**: https://sumo.dlr.de/docs/TraCI.html
- **Linux tc**: https://man7.org/linux/man-pages/man8/tc.8.html
- **HTB Queueing**: https://linux.die.net/man/8/tc-htb

