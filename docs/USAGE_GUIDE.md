# RawDataToPresentations 使用ガイド

## 🚀 クイックスタートガイド

### 1. 初回セットアップ

```bash
# リポジトリに移動
cd /Users/miura/_git/RawDataToPresentations

# 仮想環境をアクティベート
source venv/bin/activate

# 実行権限の確認
ls -la nd2_to_keynote.py
# -rwxr-xr-x であることを確認
```

### 2. 基本的な使用方法

```bash
# 現在のディレクトリのND2ファイルを自動処理
./nd2_to_keynote.py --input .

# 特定のディレクトリを処理
./nd2_to_keynote.py --input "/Volumes/Amazon/20251111_PrimaryPodocyteIHC_Hoechst_AlphaTubulin_PR"
```

## 📸 画像処理の詳細

### 蛍光チャネルの自動検出

スクリプトは以下のキーワードでチャネルを自動検出します：

| チャネル | 検出キーワード | 色 |
|---------|---------------|---|
| 核染色 | `dapi`, `hoechst`, `blue` | 青 |
| 抗体・タンパク質 | `alexa488`, `fitc`, `gfp`, `green` | 緑 |
| 細胞骨格・抗体 | `alexa568`, `tubulin`, `cy3`, `red` | 赤 |

### 実行例とログ出力

```bash
./nd2_to_keynote.py --input . --image-only --all-channels
```

**期待される出力:**
```
Found 1 ND2 file(s):
  - x20x8_02.nd2

🔍 Auto-detecting content type...
  x20x8_02.nd2: image (1 frames)

📸 Processing 1 image file(s)...
Processing x20x8_02.nd2...
  Image shape: (1024, 1024, 3)
  Final shape: (3, 1024, 1024)
  Number of channels: 3
    Channel 0: Hoechst
      → Mapped to Blue (DAPI/Hoechst)
    Channel 1: AlphaTubulin
      → Mapped to Red (Alexa568/Cy3/Tubulin)
    Channel 2: Transmitted Detector
      → Skipped (Transmitted Detector/Brightfield)
    Channel mapping: {'blue': 0, 'red': 1}
      → Extracted Blue channel (index 0)
      → Extracted Red channel (index 1)
  Saved: x20x8_02_fluorescence.jpg

🎯 Creating Keynote presentation: Images.key
✅ Image Keynote created: /path/to/Images.key
```

## 🎬 ムービー処理の詳細

### MP4変換とメタデータ

```bash
./nd2_to_keynote.py --input . --movie-only --fps 10
```

**生成されるMP4の特徴:**
- 赤黒配色（蛍光顕微鏡に適した表示）
- H.264エンコード（QuickTime互換）
- メタデータ埋め込み（元ファイルパス、サイズ、FPS）

**メタデータ例:**
```
title: experiment_001.mp4
comment: Source: /path/to/experiment_001.nd2 | Side: 315.6µm | FPS: 10.000
description: Source: /path/to/experiment_001.nd2 | Side: 315.6µm | FPS: 10.000
```

## 🔧 トラブルシューティング

### よくある問題と解決法

#### 1. "Blueチャネルしか表示されない"

**原因**: チャネルマッピングが不完全
**解決法**:
```bash
# 全チャネル強制表示
./nd2_to_keynote.py --input . --image-only --all-channels

# または特定チャネル指定
./nd2_to_keynote.py --input . --image-only --channels "red,green,blue"
```

#### 2. "ND2Reader object has no attribute 'asarray'"

**原因**: ND2Readerの使用方法の問題（修正済み）
**解決法**: 最新版では修正されています

#### 3. "No ND2 backend available"

**原因**: 必要なライブラリが未インストール
**解決法**:
```bash
source venv/bin/activate
pip install nd2reader pims opencv-python numpy pillow
```

#### 4. "AppleScript failed"

**原因**: Keynoteのアクセス権限
**解決法**:
1. システム設定 → プライバシーとセキュリティ
2. オートメーション → ターミナル/Python → Keynote を許可

### デバッグ方法

```bash
# 詳細ログで実行
./nd2_to_keynote.py --input . --verbose

# 中間ファイルを保持
./nd2_to_keynote.py --input . --image-only --keep-jpgs

# 特定の処理のみテスト
./nd2_to_keynote.py --input . --image-only  # 画像のみ
./nd2_to_keynote.py --input . --movie-only  # ムービーのみ
```

## 📊 実際の使用例

### 研究データの処理

```bash
# 実験フォルダ全体を処理
./nd2_to_keynote.py --input "/Volumes/ExperimentData/20251111_Experiment" --theme "White"

# 結果:
# - Images.key (蛍光画像のスライド)
# - Movies.key (タイムラプスムービーのスライド)
```

### カスタム設定での処理

```bash
# 高品質設定
./nd2_to_keynote.py --input . --movie-only --no-clahe --codec avc1 --norm-mode percentile

# 特定チャネルのみ
./nd2_to_keynote.py --input . --image-only --channels "blue,red" --theme "ExperimentalDataR1"
```

## 📁 ファイル出力の理解

### 生成されるファイル

```
実験フォルダ/
├── experiment_001.nd2          # 元ファイル
├── experiment_001.mp4          # 変換されたムービー（ムービーの場合）
├── experiment_001_brightfield.mp4  # 明視野チャネル（存在する場合）
├── temp_fluorescence_images/   # 中間画像ファイル（--keep-jpgsの場合）
│   └── experiment_001_fluorescence.jpg
├── Images.key                  # 画像プレゼンテーション
└── Movies.key                  # ムービープレゼンテーション
```

### Keynoteスライドの構成

**画像スライド:**
- タイトル: ファイル名（拡張子なし）
- 画像: RGB合成蛍光画像
- メタデータ: フィールドサイズ（µm/side）

**ムービースライド:**
- タイトル: フォルダ名/ファイル名
- ムービー: MP4ファイル
- メタデータ: フィールドサイズ、フレームレート、総フレーム数

## 🎯 最適な使用方法

### 推奨ワークフロー

1. **データ準備**
   ```bash
   # ND2ファイルを整理されたフォルダ構造に配置
   ExperimentData/
   ├── Condition_A/
   │   ├── sample_001.nd2
   │   └── sample_002.nd2
   └── Condition_B/
       ├── sample_003.nd2
       └── sample_004.nd2
   ```

2. **自動処理実行**
   ```bash
   ./nd2_to_keynote.py --input "ExperimentData"
   ```

3. **結果確認**
   - 生成されたKeynoteファイルを開く
   - メタデータが正しく表示されているか確認
   - 必要に応じてスライドレイアウトを調整

### パフォーマンス最適化

```bash
# 大量ファイル処理時
./nd2_to_keynote.py --input . --no-overwrite  # 既存ファイルをスキップ

# 高速処理（品質より速度重視）
./nd2_to_keynote.py --input . --movie-only --no-ffmpeg --no-clahe
```

このガイドを参考に、効率的にND2ファイルからKeynoteプレゼンテーションを作成してください。
