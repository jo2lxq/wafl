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
2. 各ノードに SSH 経由で接続
3. デプロイディレクトリを作成 (`deployment_location`)
4. プロジェクトコードと生成された設定ファイルを転送
5. 各エージェント用の `config/config.json` を生成・配置

#### 転送されるファイル

- `wafl/` ディレクトリ（全 Python コード）
- `data/` ディレクトリ（データセット，接触パターン）
- エージェント固有の設定ファイル（自動生成）

#### ログ出力例

```
📋 Deploying configurations to agent 0
🔗 Connecting to denjo@192.168.11.100 for configuration deployment
📋 Deployed agent configuration to agent 0
📋 Deployed contact pattern 'rgg_n03_a1000_d10_s01.json' to agent 0
✅ All configurations deployed successfully to agent 0 (2 files)
```

**注意**:
- デプロイは実験開始前に毎回実行すること
- 設定ファイルやコードを変更した場合は再デプロイが必要

---

### 2. 実験開始 (Start Experiment)

実験を開始する．

```bash
mise start
```

#### 自動実行される処理

1. **コンテナ起動**: 全リモートノードで Docker コンテナを起動
   ```bash
   docker run -d --name wafl-node-0 \
     -p 10001:10001 \                  # 制御ポート
     -p 10002:10002 \                  # P2P ポート
     wafl-node:latest
   ```

2. **CPU 制限の適用** (WAFL フェーズ開始時):
   ```bash
   docker update --cpus="1.0" wafl-node-0  # WAFLフェーズで動的に適用
   ```

3. **ネットワークエミュレーション**: `tc` ルールを適用
   ```bash
   tc qdisc add dev veth0 root netem \
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
⚙️  Synchronization: SSP (Threshold: 90%)
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

### 3. 結果分析 (Analyze Results)

実験終了後，全ノードからログと学習結果を収集し，グラフを生成する．

```bash
mise analyze
```

#### 実行内容

1. 各ノードから SSH 経由で結果を収集
2. 収集したデータを分析
3. 精度・損失のグラフを生成

#### 収集されるファイル

各ノードから以下が収集される：

| ファイル                  | 内容                                        | 形式       |
| ------------------------- | ------------------------------------------- | ---------- |
| `metrics_{node_id}.jsonl` | 構造化ログ（エポック完了，SSP，UDP 統計等） | JSON Lines |
| `learning-data.csv`       | 学習メトリクス（精度，損失）                | CSV        |
| `resources_{node_id}.csv` | システムリソース (CPU，メモリ，NIC)         | CSV        |
| `model_instance.pth`      | 最終モデルチェックポイント                  | PyTorch    |
| `output.log`              | 標準出力ログ                                | テキスト   |

#### 保存先

```
results/
└── {experiment_id}/
    ├── summary/               # コントロールサーバーのログ
    │   └── ctrl_output.log
    └── collected/             # 各エージェントから収集
        ├── 0/
        │   ├── metrics_0.jsonl
        │   ├── learning-data.csv
        │   └── ...
        ├── 1/
        │   ├── metrics_1.jsonl
        │   └── ...
        └── 2/
            └── ...
```

#### JSON Lines ログの例

```jsonl
{"timestamp": 1732567890.123, "node": "0", "type": "epoch_complete", "epoch": 1, "train_acc": 0.8234, "train_loss": 0.4567, "test_acc": 0.8012, "test_loss": 0.5123, "phase": "WAFL"}
{"timestamp": 1732567950.456, "node": "0", "type": "ssp_force_next", "epoch": "00005", "phase": "WAFL", "wasted_ms": 1234.56, "wasted_norm": 0.012345, "batches_processed": 18}
{"timestamp": 1732568010.789, "node": "0", "type": "udp_stats", "survival_rate": 0.98, "sent_models": 45, "fec_recovery_success": 42, "fec_recovery_fail": 3}
```

---

### 4. 高度な使用法

#### 4.1 カスタムトポロジーで実験

```bash
# トポロジー生成
python utils/generate_rgg_topology.py --nodes 5 --epochs 500 --density dense --randomseed 42

# 2. parameters.json で指定
# "contact_pattern": "rgg_n05_a1000_d10_s42.json"

# 3. 実験実行
mise deploy
mise start
```

#### 4.2 異なる SSP 閾値での比較実験

**実験 1: 厳密同期 (Baseline)**
```json
{"ssp": {"enabled": false}}
```

**実験 2: SSP 80% 閾値**
```json
{"ssp": {"enabled": true, "ssp_threshold": 0.8}}
```

**実験 3: SSP 95% 閾値**
```json
{"ssp": {"enabled": true, "ssp_threshold": 0.95}}
```

各設定で実験を実行し，結果を比較する．

#### 4.3 UDP + FEC の有効性検証

**Step 1**: Baseline (TCP，パケットロスなし)
```json
{
  "method": {"udp": {"enabled": false}},
  "network_condition": {"loss": "0%"}
}
```

**Step 2**: TCP (パケットロス 3%)
```json
{
  "method": {"udp": {"enabled": false}},
  "network_condition": {"loss": "3%"}
}
```

**Step 3**: UDP + FEC (パケットロス 3%)
```json
{
  "method": {"udp": {"enabled": true, "fec_m": 9}},
  "network_condition": {"loss": "3%"}
}
```

`results/` 内の `survival_rate` を比較する．

#### 4.4 Adaptive Compression の動作確認

```json
{
  "method": {
    "compression": {"enabled": true, "initial_method": "zlib"}
  }
}
```

実験中，`metrics_{node_id}.jsonl` で compression_method の変化を確認：
```jsonl
{"type": "compression_stats", "method": "zlib", "compression_time": 0.023, "compression_ratio": 0.31}
{"type": "compression_stats", "method": "lz4", "compression_time": 0.005, "compression_ratio": 0.52}
```

---

### 5. トラブルシューティング

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
# 1. クリーンアップ実行
# Ctrl+C で停止

# 2. 手動で確認
sudo ss -tulpn | grep 10001

# 3. プロセスを停止
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
   # 存在しない場合は bash ctrl/setup.sh を再実行
   ```

#### 問題: ログが収集されない

**症状**: `mise analyze` 実行後，`results/` が空

**原因**:
- 実験が正常に実行されていない
- ログファイルのパスが間違っている

**確認方法**:
```bash
# 実行サーバー上でログを直接確認
ssh denjo@192.168.11.100 "ls -lh /home/denjo/workspace/ktakahashi/results/*/node_0/metrics_0.jsonl"
```

---

### 6. ベストプラクティス

#### 実験前チェックリスト

- [ ] `execution_config.json` の IP アドレスが正しい
- [ ] `parameters.json` の contact_pattern ファイルが存在する
- [ ] 全ノードに SSH でパスワードなし接続可能
- [ ] 全ノードに Docker イメージ `wafl-node:latest` が存在

#### 実験中の推奨事項

- ターミナルを閉じない（ログを確認し続ける）
- CPU とネットワーク使用率をモニタリング (`htop`, `iftop`)
- 異常な動作があれば早めに `Ctrl+C` で停止

#### 実験後の推奨事項

- 必ず `mise analyze` で結果を収集・分析
- 実験 ID をメモ（後で結果を特定するため）

---

## English Version

### Basic Workflow

```
Prepare → Deploy → Run Experiment → Collect Results → Cleanup
```

### 1. Deploy

```bash
mise deploy
```

Distributes latest code and configuration to all Execution Servers.

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

### 3. Collect Results

```bash
mise analyze
```

Collects logs and metrics from all nodes to `results/{experiment_id}/`.

### Troubleshooting

See Japanese version for detailed troubleshooting guide.
