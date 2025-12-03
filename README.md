# RawDataToPresentations

ND2ファイルから Keynote プレゼンテーションを自動生成するツール群

## 概要

このツール群は、Nikon ND2 ファイルを解析し、内容に応じて画像またはムービーとして処理し、自動的に Keynote プレゼンテーションを生成します。

### 主な機能

- **自動コンテンツ検出**: ND2ファイルが画像（単一フレーム）、ムービー（時系列）、ボリューム（Z-stack）かを自動判別
- **蛍光チャネル抽出**: DAPI/Hoechst（青）、Alexa488/FITC（緑）、Alexa568/Tubulin（赤）の自動マッピング
- **高品質MP4変換**: 赤黒配色、CLAHE強調、メタデータ埋め込み
- **ボリュームデータ可視化**: Z-stackムービー、XZ/YZ断面画像、物理サイズ表示
- **Keynoteプレゼンテーション自動生成**: スライドレイアウト、メタデータ表示
- **再帰的ディレクトリスキャン**: サブフォルダ内のND2ファイルを自動検出

## ツール構成

### 🎯 統合ツール

#### `nd2_to_keynote.py` - **推奨メインツール**
ND2ファイルの内容を自動判別し、適切な処理を行う統合スクリプト

```bash
# 基本使用（自動検出: image/movie/volume）
./nd2_to_keynote.py --input "."

# 画像のみ強制処理
./nd2_to_keynote.py --input "." --image-only --all-channels

# ムービーのみ強制処理  
./nd2_to_keynote.py --input "." --movie-only --fps 10

# ボリュームのみ強制処理
./nd2_to_keynote.py --input "." --volume-only
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

### 📦 ボリューム処理ツール

#### `nd2volumes_to_keynote.py`
ND2ボリュームファイル（Z-stack）からKeynoteプレゼンテーションを作成

**機能:**
- Z-stackをムービーとして再生（Z軸を時間軸として表示）
- XZ断面画像（中央Y位置での断面）
- YZ断面画像（中央X位置での断面）
- 物理サイズ（X, Y, Z in µm）をスライドに表示
- カラーチャンネル自動検出・マッピング

```bash
# 基本使用
python3 nd2volumes_to_keynote.py --input "volume.nd2"

# ディレクトリ内のND2ファイルを処理
python3 nd2volumes_to_keynote.py --input "/path/to/dir"

# 詳細出力モード
python3 nd2volumes_to_keynote.py --input "volume.nd2" --verbose

# 一時ファイルを保持
python3 nd2volumes_to_keynote.py --input "volume.nd2" --keep-temp

# カスタム設定
python3 nd2volumes_to_keynote.py --input "volume.nd2" --fps 15 --theme "Black"
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

#### ボリューム処理オプション
- `--fps FLOAT`: Z-stackムービーのフレームレート（デフォルト: 10）
- `--clip-limit FLOAT`: CLAHEクリップリミット（デフォルト: 3.0）
- `--tile-grid INT`: CLAHEタイルグリッドサイズ（デフォルト: 8）
- `--keep-temp`: 一時ファイル（MP4、JPG）を保持
- `--verbose`: 詳細出力（チャンネル検出情報など）

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
├── nd2volumes_to_keynote.py   # ボリューム処理専用 🆕
├── nd2_to_mp4.py             # MP4変換専用（--green-only対応）
├── nd2_utils.py              # 共通ユーティリティ関数 🆕
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

### v2.2 (2024-12-03) 🆕
**ボリュームデータ対応 & コードリファクタリング**

#### 新機能
- **`nd2volumes_to_keynote.py`**: Z-stackボリュームデータ用の新スクリプト追加
  - Z-stack ムービー生成（Z軸を時間として再生）
  - XZ断面画像（中央Y位置での断面）
  - YZ断面画像（中央X位置での断面）
  - ボリューム物理サイズ表示（X, Y, Z in µm）
  - 3要素を512x512にスケーリング、位置(30,180)に配置
  
- **`nd2_to_keynote.py`にボリューム検出機能追加**
  - image / movie / volume の3タイプを自動検出
  - `--volume-only` オプション追加
  - `--keep-temp` オプション追加（一時ファイル保持）

- **`nd2_utils.py`**: 共通ユーティリティモジュール新規作成
  - バックエンド検出: `check_nd2_backends()`, `require_nd2_backend()`, `require_cv2()`
  - ファイル検索: `find_nd2_files_recursively()`, `find_nd2_files_in_directory()`
  - メタデータ: `metadata_to_dict()`, `detect_fps_from_metadata()`, `extract_side_length_um()`, `get_pixel_size_um()`
  - チャンネル検出: `classify_channel_names()`, `print_channel_info()`
  - 画像処理: `to_native_endian()`, `normalize_to_uint8()`, `apply_clahe_if_needed()`, `build_rgb_from_channels()`, `create_colored_frame()`
  - その他: `check_ffmpeg_availability()`, `detect_nd2_content_type()`

#### 改善
- **カラーチャンネル検出の改善**
  - `nd2images_to_keynote.py`: チャンネル検出結果を詳細に標準出力に表示（🔴🟢🔵絵文字付き）
  - `nd2volumes_to_keynote.py`: 単一チャンネル時もch_mapに基づいて正確な色で表示（緑固定→検出色）
  
- **コードリファクタリング**
  - 全スクリプト（nd2_to_mp4.py, nd2images_to_keynote.py, nd2movies_to_keynote.py, nd2volumes_to_keynote.py, nd2_to_keynote.py）が`nd2_utils.py`を使用するように更新
  - 重複コードの削減、メンテナンス性向上
  - 下位互換性を維持（nd2_utilsがない場合はフォールバック）

#### 使用例
```bash
# ボリュームデータの処理
python nd2volumes_to_keynote.py --input "volume.nd2" --verbose

# 自動検出（image/movie/volume）
python nd2_to_keynote.py --input "." 

# ボリュームのみ強制処理
python nd2_to_keynote.py --input "." --volume-only
```

### v2.1
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

**推奨ワークフロー**: `nd2_to_keynote.py` → 自動検出（image/movie/volume）→ Keynoteプレゼンテーション完成 🎉

**出力ファイル**:
- `*_Images.key`: 画像データ用
- `*_Movies.key`: ムービーデータ用  
- `*_Volume.key`: ボリュームデータ用（各ファイルごとに生成）
