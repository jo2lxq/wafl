# セットアップガイド / Setup Guide

**日本語** | [English](#english-version)

---

## 日本語版

### 前提条件

#### コントロールサーバー
- **OS**: Linux (Ubuntu 20.04 LTS 以降を推奨する)
- **Python**: 3.11 以上
- **ネットワーク**: 全ての実行サーバーへ SSH 接続が可能であること
- **ディレクトリ**: 任意の作業ディレクトリ（Docker のインストールは不要である）

#### 実行サーバー（エージェント）
- **OS**: Linux (Ubuntu 20.04 LTS 以降を推奨する)
- **Docker**: インストール済みであり，バックグラウンドでデーモンが実行中であること
- **権限**: Docker コマンドを非 root ユーザーで実行可能であること（docker グループへの所属）
- **ポート**: 10001 〜 13999 の範囲が利用可能であること（設定により変動する）

#### ネットワーク要件
- コントロールサーバーから全ての実行サーバーに対し，公開鍵認証による**パスワードなし SSH アクセス**が可能であること

---

### セットアップ手順

#### ステップ 1: mise のインストール（コントロールサーバー）

`mise` はランタイム管理およびタスク実行のためのツールである．

```bash
# 1. mise をインストールする
curl https://mise.run | sh

# 2. シェルに mise を登録する
# bash の場合:
echo 'eval "$(~/.local/bin/mise activate bash)"' >> ~/.bashrc
source ~/.bashrc

# zsh の場合:
echo 'eval "$(~/.local/bin/mise activate zsh)"' >> ~/.zshrc
source ~/.zshrc
```

**確認**:
```bash
mise --version
```

#### ステップ 2: プロジェクトのセットアップ（コントロールサーバー）

```bash
# 1. プロジェクトディレクトリへ移動する
cd WAFL-Testbed

# 2. 依存関係の自動インストールを実行する
mise setup
```

`mise setup` コマンドは以下の処理を自動的に実行する．
- Python 3.11 のインストール
- 高速なパッケージ管理ツール `uv` の導入
- 仮想環境 `.venv` の構築
- `pyproject.toml` に定義された全ての Python 依存ライブラリのインストール
- 開発用 pre-commit フックの登録

**確認**:
```bash
# Python バージョンの確認
python --version  # Python 3.11.x 以上であることを確認する

# インストール済みパッケージの確認
uv pip list
```

#### ステップ 3: SSH 鍵の設定（コントロールサーバー）

実行サーバー群に対し，パスワード入力を省略して SSH 接続できるように設定する．

```bash
# 1. SSH キーペアの生成（未作成の場合のみ）
ssh-keygen -t ed25519 -C "wafl-testbed"
# 全てのプロンプトで Enter を押し，デフォルト設定を適用する

# 2. 公開鍵を実行サーバーへ配布する
# 例: ノード 0 (192.168.11.100)
ssh-copy-id -i ~/.ssh/id_ed25519.pub denjo@192.168.11.100

# 実行サーバーの IP アドレスごとに上記を繰り返す
```

**確認**:
```bash
# パスワードなしでリモートコマンドが実行できることを確認する
ssh denjo@192.168.11.100 "echo 'Successfully connected via SSH'"
```

**トラブルシューティング**:
- `Permission denied (publickey)`: 公開鍵が正しくコピーされていない可能性
  - `ssh-copy-id` を再実行
  - または手動で `~/.ssh/authorized_keys` に追加
- `Connection refused`: SSH デーモンが起動していない
  - 実行サーバーで `sudo systemctl start sshd` を実行

#### ステップ 4: ノードセットアップ（全実行サーバー）

全実行サーバーを一括でセットアップする統合スクリプトを実行する．

```bash
# コントロールサーバーから実行
bash ctrl/setup.sh
```

**このスクリプトが自動的に行うこと**:

1. **Sudo 権限設定**: パスワード不要で sudo を実行できるように設定
2. **パッケージインストール**: Docker, Chrony, sysstat, jq, rsync 等をインストール
3. **ホスト設定**: 
   - Chrony (時刻同期) の設定
   - Docker グループへのユーザー追加
   - Kernel パラメータのチューニング

**実行時の入力要求**:
- SSH パスワード: 各実行サーバーへの SSH 接続用
- Sudo パスワード: 実行サーバーでの sudo コマンド実行用

**注意**: 
- 初回セットアップ時のみ実行すること
- Docker イメージのビルドと配布は `mise deploy` で行われるため、このスクリプトでは行われない
- `ctrl/execution_config.json` にノード情報が正しく設定されていることを確認すること
- `jq` と `sshpass` がローカルにインストールされている必要がある

```bash
# 必要に応じてインストール
sudo apt-get install -y jq sshpass
```

#### ステップ 5: 設定ファイルの編集（コントロールサーバー）

実験構成およびパラメータを自環境に合わせて調整する．

**`ctrl/execution_config.json`** - インフラ構成定義:

```json
{
  "nodes": [
    {
      "name": 0,
      "physical_ip": "192.168.11.100",  // 実際の IP アドレスへ変更する
      "container_port_ctrl": 10001,
      "host_port_ctrl": 10001,
      "host_port_p2p": 10002,
      "cpu_limit": "1.0"
    }
  ],
  "deployment_location": "/home/denjo/wafl",  // デプロイ先パス
  "user": "denjo"  // SSH ユーザー名
}
```

**`ctrl/parameters.json`** - 実験パラメータ定義:

```json
{
  "epochs": {
    "self": 64,
    "wafl": 4096
  },
  "contact_pattern": "rgg_n03_a1000_d10_s01.json",
  "wafl_phase": {
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "method": "udp",
  "ssp": {
    "enabled": true,
    "ssp_threshold": 0.8
  }
}
```

#### ステップ 6: データセットとトポロジーの生成（コントロールサーバー）

**トポロジー生成**:

```bash
# RGG Dense (3 ノード，100 エポック，平均次数≥10)
python utils/generate_rgg_topology.py --nodes 3 --epochs 100 --density dense --randomseed 1

# RGG Sparse (3 ノード，100 エポック，平均次数≤4)
python utils/generate_rgg_topology.py --nodes 3 --epochs 100 --density sparse --randomseed 1

# RWP (移動あり)
python utils/generate_rwp_topology.py --nodes 3 --times 100
```

生成されたファイルは `data/contact_pattern/` に保存される．

**データセット生成（初回のみ）**:

```bash
# Non-IID MNIST データセット生成
python utils/generate_datasets.py

# Non-IID フィルター生成
python utils/generate_nonIID_filters.py --ratio 50
```

#### ステップ 7: 疎通確認テスト

小規模な設定（2〜3 ノード，10 エポック程度）で動作確認を行う．

```bash
# デプロイ
mise deploy

# 実験開始（Ctrl+C で停止可能）
mise start
```

**期待される出力**:
```
🚀 Starting experiment: wafl-experiment-20251125T195801
📋 Phase 0: Creating agents and deploying configurations
✅ All agents created and configured successfully
🏃 Phase 1: Starting SELF phase (64 epochs)
✅ Agent 0 completed SELF epoch 00001
✅ Agent 1 completed SELF epoch 00001
✅ Agent 2 completed SELF epoch 00001
...
```

---

### トラブルシューティング

#### 問題: `docker: command not found`

**原因**: 実行サーバーに Docker がインストールされていない

**解決方法**:
```bash
# Ubuntu/Debian の場合
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# ユーザーを docker グループに追加（再ログイン必要）
sudo usermod -aG docker $USER
```

#### 問題: `Permission denied` (Docker)

**原因**: 現在のユーザーが docker グループに属していない

**解決方法**:
```bash
sudo usermod -aG docker $USER
# ログアウト→ログインして適用
```

#### 問題: ポート衝突 (`Address already in use`)

**原因**: 既に使用中のポートが設定されている

**解決方法**:
```bash
# 使用中のポートを確認
sudo ss -tulpn | grep 10001

# 既存プロセスを停止
# Ctrl+C で停止するか，手動で docker stop を実行

# または execution_config.json でポート番号を変更
```

#### 問題: SSH 接続タイムアウト

**原因**: ネットワーク設定，ファイアウォール

**解決方法**:
```bash
# ping で疎通確認
ping 192.168.11.100

# SSH ポートの確認（デフォルト 22）
telnet 192.168.11.100 22

# ファイアウォールの確認 (Ubuntu)
sudo ufw status
sudo ufw allow 22/tcp
```

---

### 関連ドキュメント

- [システムアーキテクチャ](architecture.md) - 設計とコンポーネント
- [設定ガイド](configuration.md) - パラメータ詳細
- [使用方法](usage.md) - 実験実行手順
- [通信プロトコル詳細](protocol.md) - TCP / UDP / Fast モードの実装詳細
- [結果分析ガイド](analysis.md) - グラフ・レポートの解釈方法

---

## English Version

### Prerequisites

**Control Server**:
- OS: Linux (Ubuntu 20.04 LTS+ recommended)
- Python: 3.11+
- Network: SSH access to all Execution Servers
- Directory: Any working directory (Docker not required)

**Execution Servers (Agents)**:
- OS: Linux (Ubuntu 20.04 LTS+ recommended)
- Docker: Installed & running
- Permissions: Non-root Docker execution (member of docker group)
- Ports: 10001〜13999 available (depending on configuration)

**Network Requirements**:
- **Passwordless SSH access** from Control Server to all Execution Servers (public key authentication)

### Setup Steps

#### Step 1: Install mise (Control Server)

```bash
curl https://mise.run | sh
echo 'eval "$(~/.local/bin/mise activate bash)"' >> ~/.bashrc
source ~/.bashrc
```

#### Step 2: Project Setup

```bash
cd WAFL-Testbed
mise setup
```

#### Step 3: SSH Configuration

```bash
ssh-keygen -t ed25519 -C "wafl-testbed"
ssh-copy-id denjo@192.168.11.100
# Repeat for all nodes
```

#### Step 4: Node Setup

Run the unified setup script to configure all execution servers.

```bash
bash ctrl/setup.sh
```

This script automatically:
1. Configures passwordless sudo
2. Installs packages (Docker, Chrony, sysstat, jq, rsync, etc.)
3. Configures hosts (time sync, Docker permissions, kernel tuning)

**Note**: Docker image build and distribution are handled by `mise deploy`.

**Prerequisites**: `jq` and `sshpass` must be installed locally.

```bash
sudo apt-get install -y jq sshpass
```

#### Step 5: Edit Configuration Files

Edit `ctrl/execution_config.json` and `ctrl/parameters.json` according to your environment.

#### Step 6: Generate Datasets and Topology

```bash
python utils/generate_rgg_topology.py --nodes 3 --epochs 100 --density dense
python utils/generate_datasets.py
```

#### Step 7: Verification Test

```bash
mise deploy
mise start
```

### Troubleshooting

See Japanese version for detailed troubleshooting guide.
