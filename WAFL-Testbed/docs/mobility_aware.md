# モビリティ対応ネットワークエミュレーション / Mobility-Aware Network Emulation

**日本語** | [English](#english-version)

---

## 日本語版

### 概要

従来の静的ネットワーク条件（一定の帯域・遅延・ロス率）に対し，**ノード間距離に応じて通信品質がリアルタイムに変化**する環境を実現する．これにより，実社会の移動体通信（VANET 等）におけるフェージングや切断プロセスを実機ベースで模擬できる．

#### 主な機能

- **SUMO 統合**: SUMO (Simulation of Urban MObility) シミュレーションから全車両のモビリティトレースを抽出する．
- **距離ベース品質計算**: 各エポックにおける全ノード間の三次元的な距離を算出し，通信品質（帯域・遅延・パケットロス率）を決定する．
- **Per-Peer Limitation**: Linux カーネルの HTB (Hierarchical Token Bucket) とフィルタ機能を活用し，通信相手（ピア）ごとに独立したネットワーク制限を適用する．
- **4 段階ランク制**: 距離減衰モデルに基づき Excellent / Good / Fair / Poor の 4 段階で品質を管理する．
- **設定可能な距離減衰モデル**: JSON 形式の定義ファイルにより，任意の通信環境（都市部，高速道路，建物内等）を再現可能である．
- **mise タスク統合**: `mise sumo` コマンドにより，SUMO の実行からトレース生成，可視化までを一貫して実行できる．

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

#### 2. SUMO シナリオの作成

モビリティシナリオ（`.sumocfg`，`.rou.xml`，`.net.xml`）を `data/sumo/` に配置する．

例: 簡単な交差点シナリオ

```bash
# data/sumo/ ディレクトリ構造例
data/sumo/
├── scenario.sumocfg      # SUMO 設定ファイル
├── network.net.xml       # 道路ネットワーク
└── routes.rou.xml        # 車両ルート
```

#### 3. モビリティトレースとネットワーク条件の生成

`mise sumo` コマンドは，SUMO シミュレーションを実行後，モビリティトレース，コンタクトパターン，ネットワーク条件の定義ファイル，および可視化資料を一括で生成する．

```bash
mise sumo
```

#### 内部的な実行プロセス

1. **SUMO シミュレーション実行**: `prepare_mobility.py` が TraCI (Traffic Control Interface) を介して SUMO を制御し，全ノードの座標情報を時系列で記録する．
2. **コンタクトパターン算出**: 各エポック（時刻）において無線通信が可能なノードペアを特定し，連合学習用トポロジーを構築する．
3. **動的ネットワーク条件の生成**: 距離減衰モデルに基づき，各ノードペア間の詳細な tc 設定を生成する．
4. **可視化と解析**: `visualize_sumo_results.py` を用いて，車両の動きや通信品質の変化をグラフおよび動画アニメーション（HTML 形式）として出力する．



#### 4. 距離減衰モデルのカスタマイズ

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

#### 5. 実験での利用

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
mise deploy
mise start
mise analyze
```

### OSM 実地図モード (Real-World Map Mode)

従来の格子状（マンハッタングリッド）シナリオの代わりに，OpenStreetMap (OSM) から取得した実際の道路ネットワーク上でシミュレーションを行うことが可能である．

#### OSM データの取得手順

1.  [OpenStreetMap Export](https://www.openstreetmap.org/export) にアクセスする．
2.  実験対象のエリアを選択する（計算負荷の観点から 1 km × 1 km 程度の範囲を推奨する）．
3.  **「エクスポート」** ボタンをクリックし，OSM 形式のファイルをダウンロードする．
4.  ダウンロードしたファイルを `data/osm/map.osm` としてプロジェクト内に保存する．

```bash
# 推奨ディレクトリ構造
data/osm/
└── map.osm    # osm 形式の地図データ
```

#### 変換と実行

```bash
mise sumo-osm
```

このコマンドを実行すると，内部で `netconvert` を用いて OSM データが SUMO 用のネットワークファイルへ変換され，ランダムな需要（車両ルート）が生成される．

#### 期待される効果

-   **一方通行路の影響**: 物理的な距離が極めて近くても，一方通行等のため大幅な迂回が必要となり，一時的に通信が途絶するシナリオが発生する．
-   **交差点での渋滞**: 信号待ちにより車両が密集する地点では，一時的に Dense (高密度) トポロジーが形成される．
-   **地理的偏り**: 幹線道路にトラフィックが集中し，住宅街などの細街路では接続が疎になる等，現実味のある学習環境が構築される．

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

tc コマンドはコンテナ内で `docker exec` を使用して実行される．コンテナ起動時に `--cap-add=NET_ADMIN` が付与されているため，通常は権限エラーは発生しない．

#### 動的 tc 設定が適用されない

1. `mobility_aware.enabled` が `true` か確認
2. ログファイルで "Applying dynamic network conditions" メッセージを確認
3. コンテナ内で手動確認:
   ```bash
   docker exec wafl-node-0 tc qdisc show dev eth0
   docker exec wafl-node-0 tc class show dev eth0
   docker exec wafl-node-0 tc filter show dev eth0
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
- **mise Task Integration**: Easy preprocessing execution with `mise sumo`

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

#### 2. Create SUMO Scenario

Place mobility scenarios (`.sumocfg`, `.rou.xml`, `.net.xml`) in `data/sumo/`.

Example: Simple intersection scenario

```bash
# Example data/sumo/ directory structure
data/sumo/
├── scenario.sumocfg      # SUMO configuration file
├── network.net.xml       # Road network
└── routes.rou.xml        # Vehicle routes
```

#### 3. Generate Mobility Trace and Network Conditions

Run the `mise sumo` command to execute SUMO simulation and generate mobility traces, contact patterns, network conditions, and visualizations all at once.

```bash
mise sumo
```

#### Execution Details

1. **Run SUMO Simulation**: Generate mobility traces with `prepare_mobility.py`
2. **Calculate Contact Patterns**: Determine communicable pairs based on inter-node distances
3. **Generate Network Conditions**: Create tc settings based on distances
4. **Visualization**: Generate graphs and animations with `visualize_sumo_results.py`

#### 4. Customize Path Loss Model

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

#### 5. Use in Experiments

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
mise deploy
mise start
mise analyze
```

### OSM Real-World Map Mode

Instead of Manhattan grid, simulate on **real road networks from OpenStreetMap data**.

#### Obtaining OSM Data

1. Access [OpenStreetMap Export](https://www.openstreetmap.org/export)
2. Select desired area (recommend ~1km × 1km)
3. Click **"Export"**
4. Save downloaded file as `data/osm/map.osm`

```bash
# Directory structure
data/osm/
└── map.osm    # Exported map file
```

#### Execution

```bash
mise sumo-osm
```

Or run directly:

```bash
python utils/prepare_mobility.py \
  --config ctrl/execution_config.json \
  --osm data/osm/map.osm \
  --output-dir data/sumo_real/ \
  --epochs 2112
```

#### Expected Effects

- **One-way street impact**: Detours required even for nearby nodes, causing disconnections
- **Intersection congestion**: Signal waiting creates Dense states with clustered vehicles
- **Geographic bias**: Vehicles concentrate on main roads, side streets become sparse

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

tc commands are executed inside the container using `docker exec`. Since containers are started with `--cap-add=NET_ADMIN`, permission errors should not normally occur.

#### Dynamic tc Configuration Not Applied

1. Confirm `mobility_aware.enabled` is `true`
2. Check for "Applying dynamic network conditions" messages in log files
3. Manually verify inside container:
   ```bash
   docker exec wafl-node-0 tc qdisc show dev eth0
   docker exec wafl-node-0 tc class show dev eth0
   docker exec wafl-node-0 tc filter show dev eth0
   ```

### References

- **SUMO Official**: https://sumo.dlr.de/
- **TraCI (Python API)**: https://sumo.dlr.de/docs/TraCI.html
- **Linux tc**: https://man7.org/linux/man-pages/man8/tc.8.html
- **HTB Queueing**: https://linux.die.net/man/8/tc-htb

---

### 関連ドキュメント / Related Documents

- [システムアーキテクチャ / System Architecture](architecture.md)
- [設定ガイド / Configuration Guide](configuration.md)
- [通信プロトコル詳細 / Protocol Details](protocol.md)
- [結果分析ガイド / Analysis Guide](analysis.md)

