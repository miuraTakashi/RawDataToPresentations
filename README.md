# RawDataToPresentations

ND2ファイルから Keynote プレゼンテーションを自動生成するツール群

## 概要

このツール群は、Nikon ND2 ファイルを解析し、内容に応じて画像またはムービーとして処理し、自動的に Keynote プレゼンテーションを生成します。

### 主な機能

- **自動コンテンツ検出**: ND2ファイルが画像（単一フレーム）かムービー（複数フレーム）かを自動判別
- **蛍光チャネル抽出**: DAPI/Hoechst（青）、Alexa488/FITC（緑）、Alexa568/Tubulin（赤）の自動マッピング
- **高品質MP4変換**: 赤黒配色、CLAHE強調、メタデータ埋め込み
- **Keynoteプレゼンテーション自動生成**: スライドレイアウト、メタデータ表示
- **再帰的ディレクトリスキャン**: サブフォルダ内のND2ファイルを自動検出

## ツール構成

### 🎯 統合ツール

#### `nd2_to_keynote.py` - **推奨メインツール**
ND2ファイルの内容を自動判別し、適切な処理を行う統合スクリプト

```bash
# 基本使用（自動検出）
./nd2_to_keynote.py --input "."

# 画像のみ強制処理
./nd2_to_keynote.py --input "." --image-only --all-channels

# ムービーのみ強制処理  
./nd2_to_keynote.py --input "." --movie-only --fps 10
```

### 📸 画像処理ツール

#### `nd2images_to_keynote.py`
ND2ファイルから蛍光チャネルを抽出してKeynoteプレゼンテーションを作成

```bash
# 全チャネル自動抽出
python3 nd2images_to_keynote.py --input "/path/to/nd2_files" --all-channels

# 特定チャネルのみ
python3 nd2images_to_keynote.py --input "." --channels "red,green,blue"

# 単一チャネル（グレースケール）
python3 nd2images_to_keynote.py --input "." --channels "blue" --single-channel
```

### 🎬 ムービー処理ツール

#### `nd2movies_to_keynote.py`
ND2ムービーファイルをMP4に変換してKeynoteプレゼンテーションを作成

```bash
# 基本変換
python3 nd2movies_to_keynote.py --input "." --fps 10

# 高品質設定
python3 nd2movies_to_keynote.py --input "." --no-clahe --codec avc1
```

### 🔬 Keyence顕微鏡対応ツール

#### `keyenceTIF_to_keynote.py`
Keyence顕微鏡のTIFFファイルからKeynoteプレゼンテーションを作成

**特徴:**
- ファイル名またはフォルダ名から倍率を自動検出（x4, x10, x20, x40）
- 検出された倍率に基づいて一辺の長さを自動計算してスライドに表示
- 倍率と一辺の長さの対応:
  - x4: 3490 µm
  - x10: 1454 µm
  - x20: 711 µm
  - x40: 360 µm

```bash
# 基本変換（倍率自動検出）
python3 keyenceTIF_to_keynote.py --input "/path/to/tiff_files"

# 画像情報も表示
python3 keyenceTIF_to_keynote.py --input "." --show-image-info
```

#### `nd2_to_mp4.py`
ND2ファイルを高品質MP4に変換（メタデータ埋め込み対応）

```bash
# 基本変換
python3 nd2_to_mp4.py --input "/path/to/nd2_files"

# パーセンタイル正規化調整
python3 nd2_to_mp4.py --input "." --lower-pct 0.5 --upper-pct 99.8

# Greenチャネルのみ出力（グレースケールMP4）
python3 nd2_to_mp4.py --input "." --green-only
```

## インストール

### 1. 依存関係のインストール

```bash
# 仮想環境の作成（推奨）
python3 -m venv venv
source venv/bin/activate

# 必要パッケージのインストール
pip install nd2reader pims opencv-python numpy pillow

# または requirements.txt を使用
pip install -r requirements.txt
```

### 2. システム要件

- **macOS** with Keynote installed
- **Python 3.8+**
- **FFmpeg** (推奨、高品質MP4エンコード用)

```bash
# FFmpeg インストール (macOS)
brew install ffmpeg
```

## 使用方法

### 🚀 クイックスタート

```bash
# 1. 仮想環境をアクティベート
source venv/bin/activate

# 2. 統合ツールで自動処理
./nd2_to_keynote.py --input "/path/to/nd2_files"

# 3. 生成されたKeynoteファイルを確認
# - Images.key (画像ファイルがある場合)
# - Movies.key (ムービーファイルがある場合)
```

詳細な使用方法については、[docs/USAGE_MANUAL.md](docs/USAGE_MANUAL.md)を参照してください。

### 📋 主要オプション（抜粋）

#### 共通オプション
- `--input DIR`: 処理対象ディレクトリ
- `--theme THEME`: Keynoteテーマ名（デフォルト: White）
- `--output FILE`: 出力ファイル名
- `--non-recursive`: サブディレクトリを検索しない
- `--verbose`: 詳細ログ表示

#### 画像処理オプション
- `--all-channels`: 全チャネルを強制抽出
- `--channels LIST`: 特定チャネル指定（例: "red,green,blue"）
- `--single-channel`: 単一チャネル（グレースケール）
- `--keep-jpgs`: 中間JPGファイルを保持

#### ムービー処理オプション
- `--fps FLOAT`: フレームレート（デフォルト: 10）
- `--no-clahe`: CLAHE強調を無効化
- `--codec CODEC`: MP4コーデック（デフォルト: avc1）
- `--norm-mode MODE`: 正規化モード（percentile/minmax）
- `--no-ffmpeg`: OpenCV VideoWriterを使用
- `--green-only`: GreenチャネルのみをグレースケールMP4として出力（出力ファイル名: `{base}_green.mp4`）

詳細なオプション一覧、使用例、トラブルシューティングについては[docs/USAGE_MANUAL.md](docs/USAGE_MANUAL.md)を参照してください。

## 主な機能の概要

- **蛍光チャネル自動検出**: DAPI/Hoechst（青）、Alexa488/FITC（緑）、Alexa568/Cy3（赤）を自動マッピング
- **MP4メタデータ埋め込み**: 元ファイルパス、フィールドサイズ、フレームレートを自動記録
- **Keyence顕微鏡対応**: 倍率自動検出（x4, x10, x20, x40）と一辺の長さ自動計算

詳細については[docs/USAGE_MANUAL.md](docs/USAGE_MANUAL.md)を参照してください。

## ファイル構成

```
RawDataToPresentations/
├── nd2_to_keynote.py          # 統合メインツール ⭐
├── nd2images_to_keynote.py    # 画像処理専用
├── nd2movies_to_keynote.py    # ムービー処理専用
├── nd2_to_mp4.py             # MP4変換専用（--green-only対応）
├── keyenceTIF_to_keynote.py   # Keyence顕微鏡TIFF処理（倍率自動検出）
├── mp4_to_keynote.py         # MP4からKeynote作成
├── requirements.txt          # Python依存関係
├── install.sh               # 自動インストールスクリプト
└── docs/                    # ドキュメント
    ├── README.md            # 詳細なREADME（このファイルの詳細版）
    ├── USAGE_MANUAL.md      # 詳細使用マニュアル
    ├── USAGE_GUIDE.md       # 実践的使用ガイド
    ├── nd2_to_mp4_manual.md # MP4変換詳細マニュアル
    └── KeyenceCalibration.md # Keyence顕微鏡キャリブレーション情報
```

## バージョン履歴

### v2.1 (最新)
- `nd2_to_mp4.py`に`--green-only`オプション追加（GreenチャネルのみをグレースケールMP4として出力）
- `keyenceTIF_to_keynote.py`に倍率自動検出機能追加（x4, x10, x20, x40倍レンズでの一辺の長さを自動計算してスライドに表示）

### v2.0
- 統合ツール `nd2_to_keynote.py` 追加
- 自動コンテンツ検出機能
- 蛍光チャネル自動マッピング拡張
- MP4メタデータ埋め込み機能

### v1.6
- MP4メタデータ自動埋め込み
- QuickTime互換性向上
- FFmpeg高品質エンコード

### v1.5
- 16bit画像エンディアン対応
- パーセンタイル正規化
- CLAHE強調オプション

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

バグレポート、機能要望、プルリクエストを歓迎します。

## 関連ドキュメント

- **[docs/USAGE_MANUAL.md](docs/USAGE_MANUAL.md)**: 詳細な使用方法、オプション一覧、トラブルシューティング
- **[docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)**: 実践的な使用ガイドと例
- **[docs/nd2_to_mp4_manual.md](docs/nd2_to_mp4_manual.md)**: MP4変換の詳細マニュアル

---

**推奨ワークフロー**: `nd2_to_keynote.py` → 自動検出 → Keynoteプレゼンテーション完成 🎉
