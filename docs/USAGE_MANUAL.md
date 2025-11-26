# 使用マニュアル

このドキュメントは、RawDataToPresentationsツール群の詳細な使用方法を説明します。

## 目次

1. [詳細オプション](#詳細オプション)
2. [蛍光チャネル自動検出](#蛍光チャネル自動検出)
3. [MP4メタデータ](#mp4メタデータ)
4. [高度な使用例](#高度な使用例)
5. [トラブルシューティング](#トラブルシューティング)
6. [Keyence顕微鏡対応](#keyence顕微鏡対応)

---

## 詳細オプション

### 共通オプション

| オプション | 説明 | 例 |
|---|---|---|
| `--input DIR` | 処理対象ディレクトリ | `--input "/path/to/nd2_files"` |
| `--theme THEME` | Keynoteテーマ名（デフォルト: White） | `--theme "ExperimentalDataR1"` |
| `--output FILE` | 出力ファイル名 | `--output "MyPresentation.key"` |
| `--non-recursive` | サブディレクトリを検索しない | `--non-recursive` |
| `--verbose` | 詳細ログ表示 | `--verbose` |

### 画像処理オプション

| オプション | 説明 | 例 |
|---|---|---|
| `--all-channels` | 全チャネルを強制抽出 | `--all-channels` |
| `--channels LIST` | 特定チャネル指定 | `--channels "red,green,blue"` |
| `--single-channel` | 単一チャネル（グレースケール） | `--single-channel` |
| `--keep-jpgs` | 中間JPGファイルを保持 | `--keep-jpgs` |
| `--debug-channels` | 各チャネルを個別画像として保存（デバッグ用） | `--debug-channels` |

### ムービー処理オプション

| オプション | 説明 | 例 |
|---|---|---|
| `--fps FLOAT` | フレームレート（デフォルト: 10） | `--fps 15.0` |
| `--no-clahe` | CLAHE強調を無効化 | `--no-clahe` |
| `--clip-limit FLOAT` | CLAHE clip limit（デフォルト: 3.0） | `--clip-limit 2.0` |
| `--tile-grid INT` | CLAHE tile grid size（デフォルト: 8） | `--tile-grid 16` |
| `--codec CODEC` | MP4コーデック（デフォルト: avc1） | `--codec H264` |
| `--norm-mode MODE` | 正規化モード（percentile/minmax） | `--norm-mode percentile` |
| `--lower-pct FLOAT` | パーセンタイル正規化の下限（デフォルト: 1.0） | `--lower-pct 0.5` |
| `--upper-pct FLOAT` | パーセンタイル正規化の上限（デフォルト: 99.5） | `--upper-pct 99.8` |
| `--no-ffmpeg` | OpenCV VideoWriterを使用 | `--no-ffmpeg` |
| `--green-only` | GreenチャネルのみをグレースケールMP4として出力（出力ファイル名: `{base}_green.mp4`） | `--green-only` |
| `--no-overwrite` | 既存のMP4ファイルをスキップ | `--no-overwrite` |

---

## 蛍光チャネル自動検出

### サポートする蛍光色素

| 色 | 検出キーワード | 用途 |
|---|---|---|
| **青 (Blue)** | DAPI, Hoechst, Blue | 核染色 |
| **緑 (Green)** | Alexa488, FITC, GFP, Green, EGFP | 抗体、タンパク質 |
| **赤 (Red)** | Alexa568, Cy3, Rhodamine, Tubulin, Red | 抗体、細胞骨格 |

### チャネルマッピング例

```
Channel 0: Hoechst → Blue (DAPI/Hoechst)
Channel 1: AlphaTubulin → Red (Alexa568/Cy3/Tubulin)
Channel 2: Transmitted Detector → Skipped (Brightfield)
```

### チャネル検出の動作

1. **メタデータから検出**: ND2ファイルのメタデータに含まれるチャネル名を解析
2. **キーワードマッチング**: チャネル名に含まれるキーワードから色を判定
3. **フォールバック**: 検出できない場合は、デフォルトマッピングを使用
   - Channel 0 → Blue (DAPI)
   - Channel 1 → Green (Alexa488)
   - Channel 2 → Red (Alexa568)

### チャネル選択の例

```bash
# 全チャネルを強制抽出
./nd2_to_keynote.py --input "." --image-only --all-channels

# 特定チャネルのみ（例: 青と赤のみ）
./nd2_to_keynote.py --input "." --channels "blue,red"

# 単一チャネル（グレースケール）
./nd2_to_keynote.py --input "." --channels "green" --single-channel
```

---

## MP4メタデータ

生成されるMP4ファイルには以下の情報が自動埋め込みされます：

```
title: filename.mp4
comment: Source: /path/to/file.nd2 | Side: 315.6µm | FPS: 10.000
description: Source: /path/to/file.nd2 | Side: 315.6µm | FPS: 10.000
```

### メタデータ確認方法

#### コマンドライン

```bash
# コメントのみ表示
ffprobe -v quiet -show_entries format_tags=comment -of default=noprint_wrappers=1:nokey=1 file.mp4

# 全メタデータ表示
ffprobe -v quiet -show_entries format_tags -of default=noprint_wrappers=1 file.mp4
```

#### GUIアプリケーション

- **VLC Media Player**: `⌘I` → メタデータタブ
- **QuickTime Player**: `⌘I` → ムービーインスペクタ
- **MediaInfo** (App Store): ファイルをドラッグ&ドロップ
- **Subler** (App Store): MP4メタデータ専用ツール

**注意**: macOSの「情報を見る」では表示されない場合があります。上記の専用アプリを使用してください。

---

## 高度な使用例

### 1. 研究用高品質設定

```bash
# 画像: 全チャネル、高解像度、中間ファイル保持
./nd2_to_keynote.py --input "." --image-only --all-channels --keep-jpgs

# ムービー: CLAHE無効、高品質コーデック、パーセンタイル正規化
./nd2_to_keynote.py --input "." --movie-only --no-clahe --codec avc1 --norm-mode percentile

# GreenチャネルのみをグレースケールMP4として出力
nd2_to_mp4.py --input "." --green-only
```

### 2. プレゼンテーション用設定

```bash
# 特定チャネルのみ、カスタムテーマ
./nd2_to_keynote.py --input "." --channels "blue,red" --theme "ExperimentalDataR1"

# ムービー: フレームレート調整、CLAHE強調
./nd2_to_keynote.py --input "." --movie-only --fps 15 --clip-limit 2.5
```

### 3. バッチ処理

```bash
# 複数ディレクトリの一括処理
for dir in /Volumes/Data/Experiment_*/; do
    ./nd2_to_keynote.py --input "$dir" --output "$(basename "$dir").key"
done

# 特定パターンのファイルのみ処理
find /path/to/data -name "*_sample_*.nd2" -exec dirname {} \; | sort -u | while read dir; do
    ./nd2_to_keynote.py --input "$dir" --output "$(basename "$dir").key"
done
```

### 4. パフォーマンス最適化

```bash
# OpenCV VideoWriterを使用（FFmpeg不要）
nd2_to_mp4.py --input "." --no-ffmpeg

# 既存ファイルをスキップ（再処理を避ける）
nd2_to_mp4.py --input "." --no-overwrite
```

---

## トラブルシューティング

### よくある問題と解決法

#### 1. チャネルが正しく表示されない

**症状**: 一部のチャネルが黒く表示される、または期待した色が表示されない

**解決策**:
```bash
# 全チャネル強制表示
./nd2_to_keynote.py --input "." --image-only --all-channels

# デバッグモードで各チャネルを個別保存
./nd2_to_keynote.py --input "." --image-only --debug-channels
```

**原因**:
- チャネル名がメタデータに含まれていない
- チャネルマッピングが正しく検出されていない
- 複数フレームが異なるチャネルとして保存されている

#### 2. ムービーが暗い/明るすぎる

**症状**: ムービーのコントラストが低い、または一部が白飛び/黒つぶれしている

**解決策**:
```bash
# パーセンタイル調整（より広い範囲を使用）
./nd2_to_keynote.py --input "." --movie-only --lower-pct 0.5 --upper-pct 99.8

# CLAHE強調を調整
nd2_to_mp4.py --input "." --clip-limit 2.0 --tile-grid 16

# CLAHEを無効化（フリッカーを減らす）
nd2_to_mp4.py --input "." --no-clahe
```

**原因**:
- 外れ値が正規化範囲に影響している
- CLAHEのパラメータが適切でない
- ムービー全体の輝度分布が不均一

#### 3. Keynote作成エラー

**症状**: AppleScriptエラー、Keynoteが起動しない

**解決策**:
```bash
# シンプルなテーマを使用
./nd2_to_keynote.py --input "." --theme "White" --verbose

# 詳細ログでエラー内容を確認
./nd2_to_keynote.py --input "." --verbose 2>&1 | tee error.log
```

**原因**:
- Keynoteへのアクセス権限がない
- カスタムテーマが存在しない
- AppleScriptの実行権限が不足している

**対処法**:
1. システム設定 → セキュリティとプライバシー → プライバシー → オートメーション
2. Keynoteのオートメーションを許可

#### 4. MP4メタデータが表示されない

**症状**: macOSの「情報を見る」でコメント欄が空欄

**解決策**:
- VLC、MediaInfo、Sublerなどの専用アプリを使用
- コマンドラインで確認:
  ```bash
  ffprobe -v quiet -show_entries format_tags=comment file.mp4
  ```

**原因**:
- macOSの「情報を見る」はFFmpegで埋め込んだメタデータを表示しない場合がある
- メタデータは正しく埋め込まれているが、表示ツールが対応していない

#### 5. Greenチャネルのみ出力したい

**解決策**:
```bash
# GreenチャネルのみをグレースケールMP4として出力
nd2_to_mp4.py --input "." --green-only

# 出力ファイル名: {base}_green.mp4
```

### エラーメッセージ対応

| エラー | 原因 | 解決法 |
|---|---|---|
| `No ND2 backend available` | nd2reader/pimsが未インストール | `pip install nd2reader pims` |
| `'ND2Reader' object has no attribute 'asarray'` | 古いバージョンの問題 | 最新版に更新済み（この問題は解決済み） |
| `FFmpeg not found` | FFmpegが未インストール | `brew install ffmpeg` (macOS) |
| `AppleScript failed` | Keynoteアクセス権限 | システム設定でオートメーション許可 |
| `ModuleNotFoundError: No module named 'cv2'` | OpenCVが未インストール | `pip install opencv-python` |
| `Keynoteでエラーが起きました: クラスdocumentを作成できません。 (-2710)` | テーマの問題 | デフォルトテーマを使用、または`--theme "White"`を指定 |

### デバッグのヒント

1. **詳細ログを有効化**: `--verbose`オプションを使用
2. **中間ファイルを保持**: `--keep-jpgs`で中間画像を確認
3. **チャネル情報を確認**: `--debug-channels`で各チャネルを個別保存
4. **メタデータを確認**: `ffprobe`でMP4メタデータを確認
5. **依存関係を確認**: `pip list`でインストール済みパッケージを確認

---

## Keyence顕微鏡対応

### `keyenceTIF_to_keynote.py`の特徴

Keyence顕微鏡のTIFFファイルからKeynoteプレゼンテーションを作成するツールです。

#### 倍率自動検出機能

ファイル名またはフォルダ名から倍率を自動検出し、対応する一辺の長さをスライドに表示します。

**倍率と一辺の長さの対応**:
- x4: 3490 µm
- x10: 1454 µm
- x20: 711 µm
- x40: 360 µm

**検出パターン**:
- ファイル名: `sample_x20_image.tif` → 711 µm / horizontal side
- フォルダ名: `x10_data/image.tif` → 1454 µm / horizontal side
- 大文字小文字を区別しない: `X40`, `x40`, `40x` など

#### 使用方法

```bash
# 基本変換（倍率自動検出）
python3 keyenceTIF_to_keynote.py --input "/path/to/tiff_files"

# 画像情報も表示（幅×高さ、モード）
python3 keyenceTIF_to_keynote.py --input "." --show-image-info

# 非再帰的検索（現在のディレクトリのみ）
python3 keyenceTIF_to_keynote.py --input "." --non-recursive

# 強制処理（TIFF以外も処理）
python3 keyenceTIF_to_keynote.py --input "." --force-process
```

#### 出力例

スライドには以下の情報が表示されます：
- **タイトル**: 元のフォルダ階層パス（例: `folder/subfolder/image`）
- **倍率情報**: 検出された倍率に基づく一辺の長さ（例: `711 µm / horizontal side`）
- **画像情報**（`--show-image-info`使用時）: 画像サイズとモード（例: `2048×1536px, RGB`）

---

## サポート

問題が発生した場合：

1. **詳細ログを確認**: `--verbose`オプションで詳細ログを確認
2. **依存関係を確認**: 依存関係が正しくインストールされているか確認
   ```bash
   pip list | grep -E "nd2reader|pims|opencv|numpy|pillow"
   ```
3. **FFmpegを確認**: FFmpegが利用可能か確認（`ffmpeg -version`）
4. **権限を確認**: macOSのセキュリティ設定でオートメーション許可を確認
5. **バージョンを確認**: 最新版を使用しているか確認

---

**関連ドキュメント**:
- [../README.md](../README.md) - 概要とクイックスタート
- [nd2_to_mp4_manual.md](nd2_to_mp4_manual.md) - MP4変換の詳細マニュアル
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - 実践的な使用ガイド

