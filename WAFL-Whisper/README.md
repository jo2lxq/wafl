# WAFL-Whisper

このプロジェクトは、Whisper モデルを使用した音声認識のための連合学習（Federated Learning）フレームワークです。

## 環境構築

### Singularity イメージ
1. Singularity イメージのビルド:

```bash
singularity build gtune_wafl.sif gtune_wafl.def
```

2. コンテナの実行:

```bash
singularity shell --nv --bind /path/to/elab_wafl:/path/to/elab_wafl --contain gtune_wafl.sif
```

3. contact_info ファイルの準備:

contact_info/以下に必要なファイルを配置

### conda

```bash
conda create -n WAFL-Whisper python=3.10 -y
conda activate WAFL-Whisper
pip install -r requirements.txt
```

## 設定ファイル

プロジェクトを実行する前に、`config.json`ファイルを作成する必要があります。以下のパラメータを含める必要があります：

```json
{
    "memo": "実験のメモ",
    "n_node": "ノード数",
    "pre_epoch": "事前学習のエポック数",
    "num_epoch": "協調学習のエポック数",
    "lr": "学習率",
    "contact_file": "通信パターンファイルのパス",
    "data_dir": "データディレクトリのパス",
    "data_name_list": ["data_dirの中で使用するフォルダのリスト"],
    "fl_coefficiency": "モデルのパラメータ合成時の係数(0以上1以下)",
    "seed": "乱数シード",
    "output_dir": "出力ディレクトリのパス",
    "train_batch_size": "学習用バッチサイズ",
    "test_batch_size": "テスト用バッチサイズ"
}
```

## データセットの準備

1. データディレクトリ構造:

```
data_dir/
    ├── dataset1/              # 各ノードごとのデータセット
    │   ├── audio/             # 音声ファイルを格納
    │   │   ├── sample1.wav    # 音声ファイル
    │   │   ├── sample2.wav
    │   │   └── ...
    │   └── script/            # テキストファイルを格納
    │       ├── sample1.txt    # 音声ファイルに対応するテキスト
    │       ├── sample2.txt
    │       └── ...
    ├── dataset2/              # 別のノードのデータセット
    │   ├── audio/
    │   └── script/
    └── test/                  # テストデータ
        ├── audio/
        └── script/
```

2. データセットの要件:

- 各データセットフォルダ（dataset1, dataset2, test）には`audio`と`script`の 2 つのサブフォルダが必要です
- `audio`フォルダには音声ファイル（.wav）を格納します
- `script`フォルダには対応するテキストファイル（.txt）を格納します
- 音声ファイルとテキストファイルは同じ名前（拡張子は異なる）で対応付けられます
  - 例：`audio/sample1.wav` ↔ `script/sample1.txt`

## 実行方法

Singularity コンテナ内で以下のコマンドを実行:

```bash
python chula_wafl_main.py
```

## 出力

実行後、以下のディレクトリ構造で結果が保存されます：

```
output_dir/
    ├── graph/              # グラフ画像
    │   └── average_cer.png # 平均CERの推移グラフ
    ├── text/               # テキスト出力
    │   └── node{0..n}/    # 各ノードの出力
    ├── model/              # 学習済みモデル
    ├── all_result.txt      # 実験設定と結果のサマリー
    └── cer_results.json    # CER結果（JSON形式）
```

### 出力ファイルの説明

1. `all_result.txt`

   - 実験設定（config.json の内容）
   - 実行日時
   - 最終的な平均 CER

2. `cer_results.json`
   - 各ノードの全エポックでの CER と平均 CER を JSON 形式で保存
   - 構造例：
     ```json
     {
       "node_results": {
         "node_0": [0.123, 0.115, 0.108, ...],  // ノード0の各エポックでのCER
         "node_1": [0.134, 0.128, 0.121, ...],  // ノード1の各エポックでのCER
         ...
       },
       "average_results": [0.129, 0.122, 0.115, ...]  // 全ノードの平均CERの推移
     }
     ```
