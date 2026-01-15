# 実験実行ガイド / Experiment Usage Guide

**日本語** | [English](#english-version)

---

## 日本語版

`mise` タスクランナーで実験のライフサイクル全体を管理する．

### 基本ワークフロー

```
準備 → デプロイ → 実験実行 → 結果分析 → クリーンアップ
```

---

### 1. デプロイ (Deploy)

最新のコードと設定ファイルを全実行サーバーに配布する．

```bash
mise deploy
```

#### 実行内容

1. `ctrl/execution_config.json` を読み込み
2. 管理サーバー上で Docker イメージをビルド
3. Docker Registry にプッシュ
4. 各ノードで Docker イメージをプル
5. 各エージェント用の設定ファイルを生成・配置

#### 転送されるファイル

- `wafl/` ディレクトリ（全 Python コード）
- `data/` ディレクトリ（データセット，接触パターン）
- エージェント固有の設定ファイル（自動生成）

#### ログ出力例

```
📦 Setting up Python environment on management server...
🏗️  Building Docker image on management server...
📤 Pushing Docker image to registry...
🚀 Pulling image and syncing files on all nodes...
🧹 Cleaning up...
🎉 Deployment completed!
```

> **Note**: デプロイは実験開始前に毎回実行すること．設定ファイルやコードを変更した場合は再デプロイが必要．

---

### 2. 実験開始 (Start Experiment)

実験を開始する．

```bash
mise start
```

このコマンドは `mise deploy` を自動実行後，実験を開始する．

#### 自動実行される処理

1. **コンテナ起動**: 全リモートノードで Docker コンテナを起動
   ```bash
   docker run -d --name wafl-node-0 \
     --cap-add=NET_ADMIN \
     -p 10001:10001 \
     -p 10002:10002 \
     wafl-node:latest
   ```

2. **CPU 制限の適用** (WAFL フェーズ開始時):
   ```bash
   docker update --cpus="1.0" wafl-node-0
   ```

3. **ネットワークエミュレーション**: `tc` ルールを適用
   ```bash
   tc qdisc add dev eth0 root netem \
     delay 50ms \
     loss 3% \
     rate 10mbit
   ```

4. **オーケストレーション**: ControlServer をローカルで起動し，実験を調整
   - SELF フェーズ → WAFL フェーズの順に実行
   - エポックごとに各エージェントの完了確認
   - SSP 有効時は閾値チェックと `FORCE_NEXT` 発行

#### モニタリング

コントロールサーバーのログがターミナルに表示される：

```
🚀 Starting experiment: wafl-experiment-20251125T195801 (SELF epochs: 64, WAFL epochs: 4096)
📋 Phase 0: Creating agents and deploying configurations
✅ All agents created and configured successfully
🏃 Phase 1: Starting SELF phase (64 epochs)
✅ Agent 0 completed SELF epoch 00001
✅ Agent 1 completed SELF epoch 00001
✅ Agent 2 completed SELF epoch 00001
...
🎉 All SELF training epochs completed successfully
🤝 Phase 2: Starting WAFL phase (4096 epochs)
⚙️  Synchronization: SSP (Threshold: 80%)
✅ Agent 0 completed WAFL epoch 00001
⚡ SSP Threshold reached for epoch 00005. Forcing slow agents to skip.
⏩ Forcing agent 2 (epoch 00004) to skip to 00005
...
```

#### 停止方法

**正常停止**: `Ctrl+C` を押す
- Control Server が全エージェントに `KILL` コマンドを送信
- 各エージェントは現在のエポックを完了してから終了
- ログとモデルチェックポイントが保存される

**強制停止**: `Ctrl+C` を 2 回押す
- 即座に全プロセスを強制終了（データ損失の可能性あり）

---

### 3. 複数実験の自動実行 (Run Experiments)

`ctrl/parameters/` ディレクトリ内の全パラメータファイルを順次実行する．

```bash
mise experiments
```

#### パラメータファイルの配置

```
ctrl/parameters/
├── exp0_1-excellent-1-tcp.json
├── exp0_1-excellent-3-dynamic.json
├── exp0_1-excellent-4-fast.json
├── exp0_2-good-1-tcp.json
├── exp0_2-good-3-dynamic.json
├── exp0_2-good-4-fast.json
├── exp0_3-fair-1-tcp.json
├── exp0_3-fair-3-dynamic.json
├── exp0_3-fair-4-fast.json
└── exp0_4-poor-1-tcp.json
```

#### パラメータファイル例

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
  "method": "dynamic",
  "ssp": {
    "enabled": true,
    "ssp_threshold": 0.8
  }
}
```

#### 実行順序

- ファイル名のアルファベット順で実行
- 命名規則: `exp{実験番号}_{条件番号}-{条件名}-{手法番号}-{手法名}.json`
- 例: `exp0_1-excellent-3-dynamic.json`
  - 実験 0: 基本比較実験
  - 条件 1: excellent (高帯域・低損失)
  - 手法 3: dynamic モード

#### ログ出力例

```
📋 Found 12 parameter files in ctrl/parameters/
🔄 Running experiment 1/12: exp0_1-excellent-1-tcp.json
...
✅ Experiment completed: exp0_1-excellent-1-tcp
🔄 Running experiment 2/12: exp0_1-excellent-3-dynamic.json
...
```

#### 実験の無効化

特定の実験を一時的にスキップする場合，`.disabled/` サブディレクトリに移動：

```bash
# 実験を無効化
mv ctrl/parameters/exp0_1-excellent-1-tcp.json ctrl/parameters/.disabled/

# 実験を有効化
mv ctrl/parameters/.disabled/exp0_1-excellent-1-tcp.json ctrl/parameters/
```

---

### 4. 結果分析 (Analyze Results)

実験終了後，全ノードからログと学習結果を収集し，グラフを生成する．

```bash
mise analyze
```

#### 実行内容

1. 各ノードから SSH 経由で結果を収集
2. ローカルに rsync でダウンロード
3. 収集したデータを分析
4. グラフと分析レポートを生成

詳細: [docs/analysis.md](analysis.md)

#### 収集されるファイル

各ノードから以下が収集される：

| ファイル                  | 内容                                        | 形式       |
| ------------------------- | ------------------------------------------- | ---------- |
| `metrics_{node_id}.jsonl` | 構造化ログ（エポック完了，SSP，UDP 統計等） | JSON Lines |
| `resources_{node_id}.csv` | システムリソース (CPU，メモリ，NIC)         | CSV        |
| `model_instance.pth`      | 最終モデルチェックポイント                  | PyTorch    |
| `output.log`              | 標準出力ログ                                | テキスト   |

#### 保存先

```
results/
└── {experiment_id}/
    ├── summary/               # コントロールサーバーのログ
    │   ├── ctrl_output.log
    │   └── metadata.jsonl
    ├── collected/             # 各エージェントから収集
    │   ├── 0/
    │   │   ├── metrics_0.jsonl
    │   │   └── ...
    │   ├── 1/
    │   │   └── ...
    │   └── 2/
    │       └── ...
    └── analysis/              # 分析結果
        ├── graphs/
        │   ├── test_accuracy.png
        │   ├── epoch_duration.png
        │   └── ...
        └── report.md
```

---

### 5. 設定検証・ベンチマーク (Verify)

実験前に設定の妥当性を検証し，インフラのベンチマークを実行する．

```bash
mise verify
```

#### 検証内容

1. **設定ファイル検証**
   - `parameters.json` の構文チェック
   - `execution_config.json` の構文チェック
   - 必須パラメータの存在確認
   - 接触パターンファイルの存在確認

2. **インフラ検証**
   - 各ノードへの SSH 接続確認
   - Docker デーモンの起動確認
   - Docker イメージの存在確認
   - ポートの空き状況確認

3. **ネットワーク検証**
   - ノード間のレイテンシ測定
   - 帯域幅測定（iperf3）
   - tc ルールの適用テスト

4. **ベンチマーク**
   - CPU ベンチマーク（stress-ng）
   - メモリベンチマーク
   - ネットワークスループットベンチマーク

#### ログ出力例

```
🔍 WAFL-Testbed Verification & Benchmark Tool
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Configuration Validation
  ✅ parameters.json: Valid
  ✅ execution_config.json: Valid
  ✅ Contact pattern file exists

🔗 Infrastructure Verification
  ✅ Node 0 (192.168.11.100): SSH OK, Docker OK
  ✅ Node 1 (192.168.11.101): SSH OK, Docker OK
  ✅ Node 2 (192.168.11.102): SSH OK, Docker OK

📊 Network Benchmarks
  Node 0 ↔ Node 1: RTT 0.5ms, Bandwidth 943 Mbps
  Node 0 ↔ Node 2: RTT 0.6ms, Bandwidth 941 Mbps
  Node 1 ↔ Node 2: RTT 0.4ms, Bandwidth 945 Mbps

✅ All verifications passed!
```

#### オプション

```bash
# 全検証を実行
python ctrl/verify.py --all

# 設定検証のみ
python ctrl/verify.py --config

# インフラ検証のみ
python ctrl/verify.py --infra

# ベンチマークのみ
python ctrl/verify.py --benchmark

# 特定ノードのみ検証
python ctrl/verify.py --nodes 0,1,2
```

---

### 6. SUMO モビリティ前処理 (SUMO Preprocessing)

SUMO シミュレーションからモビリティトレースを生成する．

```bash
mise sumo
```

詳細: [docs/mobility_aware.md](mobility_aware.md)

#### OSM 実地図モード

OpenStreetMap データを使用した実地図上でのシミュレーション．

```bash
mise sumo-osm
```

---

### 7. 高度な使用法

#### 7.1 カスタムトポロジーで実験

```bash
# トポロジー生成
python utils/generate_rgg_topology.py --nodes 50 --epochs 500 --density dense --randomseed 42

# parameters.json で指定
# "contact_pattern": "contact_pattern/rgg_n50_a1000_d10_s42.json"

# 実験実行
mise start
```

#### 7.2 異なるネットワーク条件での比較実験

**Excellent 条件** (高帯域・低損失):
```json
{
  "network_condition": {
    "enabled": true,
    "rate": "100mbit",
    "delay": "5ms",
    "loss": "0%"
  }
}
```

**Good 条件**:
```json
{
  "network_condition": {
    "enabled": true,
    "rate": "20mbit",
    "delay": "20ms",
    "loss": "1%"
  }
}
```

**Fair 条件**:
```json
{
  "network_condition": {
    "enabled": true,
    "rate": "5mbit",
    "delay": "50ms",
    "loss": "5%"
  }
}
```

**Poor 条件** (低帯域・高損失):
```json
{
  "network_condition": {
    "enabled": true,
    "rate": "1mbit",
    "delay": "100ms",
    "loss": "10%"
  }
}
```

#### 7.3 3 モード（TCP/Dynamic/Fast）の比較

1. 各手法用のパラメータファイルを作成
2. `ctrl/parameters/` に配置
3. `mise experiments` で順次実行
4. `mise analyze` で比較レポート生成

---

### 8. トラブルシューティング

#### 問題: 接続エラー

**症状**:
```
🔒 SSH connection error to agent 0: [Errno 111] Connection refused
```

**確認事項**:
1. SSH 設定
   ```bash
   ssh denjo@192.168.11.100 "echo OK"
   ```
2. `execution_config.json` の IP アドレスとユーザー名が正しいか

#### 問題: ポートの競合

**症状**:
```
docker: Error response from daemon: driver failed programming external connectivity: 
Bind for 0.0.0.0:10001 failed: port is already allocated.
```

**解決方法**:
```bash
# 使用中のポートを確認
sudo ss -tulpn | grep 10001

# 既存コンテナを停止
docker ps | grep wafl-node
docker stop {container_id}
```

#### 問題: エージェントがタイムアウト

**症状**:
```
❌ Timed out waiting for agent 2 to be ready
```

**原因と対処**:
1. コンテナ起動失敗
   ```bash
   ssh denjo@192.168.11.102 "docker ps -a | grep wafl-node"
   ssh denjo@192.168.11.102 "docker logs wafl-node-2"
   ```

2. ファイアウォールでポートがブロックされている
   ```bash
   sudo ufw allow 10001/tcp
   sudo ufw allow 10002/tcp
   ```

3. Docker イメージが存在しない
   ```bash
   ssh denjo@192.168.11.102 "docker images | grep wafl-node"
   # 存在しない場合は mise deploy を再実行
   ```

#### 問題: ログが収集されない

**症状**: `mise analyze` 実行後，`results/` が空

**確認方法**:
```bash
# 実行サーバー上でログを直接確認
ssh denjo@192.168.11.100 "ls -lh ~/results/*/collected/0/"
```

---

### 9. ベストプラクティス

#### 実験前チェックリスト

- [ ] `execution_config.json` の IP アドレスが正しい
- [ ] `parameters.json` の contact_pattern ファイルが存在する
- [ ] 全ノードに SSH でパスワードなし接続可能
- [ ] `mise verify` が成功する
- [ ] 十分なディスク容量がある（結果保存用）

#### 実験中の推奨事項

- ターミナルを閉じない（ログを確認し続ける）
- CPU とネットワーク使用率をモニタリング (`htop`, `iftop`)
- 異常な動作があれば早めに `Ctrl+C` で停止

#### 実験後の推奨事項

- 必ず `mise analyze` で結果を収集・分析
- 実験 ID をメモ（後で結果を特定するため）
- 比較実験の場合は `--compare` で比較レポート生成

---

## English Version

### Basic Workflow

```
Prepare → Deploy → Run Experiment → Analyze Results → Cleanup
```

### 1. Deploy

```bash
mise deploy
```

Builds Docker image and distributes to all Execution Servers.

### 2. Start Experiment

```bash
mise start
```

Automatically performs:
1. Container launch
2. Resource limits application
3. Network emulation (tc rules)
4. Orchestration (SELF → WAFL phases)

**Stop**: Press `Ctrl+C`

### 3. Run Multiple Experiments

```bash
mise experiments
```

Runs all parameter files in `ctrl/parameters/` sequentially.

### 4. Analyze Results

```bash
mise analyze
```

Collects logs and metrics from all nodes, generates graphs and reports.

Details: [docs/analysis.md](analysis.md)

### 5. Verify Configuration

```bash
mise verify
```

Validates configuration files, infrastructure, and runs benchmarks.

### Troubleshooting

See Japanese version for detailed troubleshooting guide.
