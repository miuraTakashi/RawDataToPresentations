# nd2_to_mp4.py マニュアル

## 概要
`nd2_to_mp4.py` は、指定ディレクトリ直下のサブフォルダにある ND2 ムービーを一括で読み込み、赤黒配色（Red-on-Black）の MP4 に変換するツールです。輝度の正規化はムービー全体に対して行われ、外れ値に強いパーセンタイル方式（既定）または従来の min-max 方式を選べます。任意で CLAHE によるコントラスト強調も可能です。

**注意**: 統合ツール `nd2_to_keynote.py` の使用を推奨します。このツールは自動的にコンテンツタイプを検出し、適切な処理を行います。

- 入力: `.nd2`（直下サブフォルダを自動走査）
- 出力: `.mp4`（各 ND2 と同じフォルダへ保存）
- 対応バックエンド: `nd2reader` 優先、無ければ `pims`

## 主な機能
- サブフォルダ内の `.nd2` を自動検出して変換
- 赤黒配色（グレイスケール→赤チャネル）
- ムービー全体の正規化（既定: パーセンタイル。従来: min-max）
- オプションで CLAHE によるコントラスト強調
- FPS は ND2 メタデータから取得（無ければ指定値）
- 16bit 画像のエンディアンをネイティブに揃えてから処理（暗化・反転を防止）
- **MP4 メタデータの自動埋め込み**（元ファイルパス、フィールドサイズ、フレームレート）

## 依存関係
```bash
pip install nd2reader pims opencv-python numpy
```

## 使い方（基本）
```bash
python3 nd2_to_mp4.py --input "/Volumes/KINGSTON/20250917PodocyteBis-T"
```
- 直下サブフォルダにある `.nd2` を走査し、各フォルダに `.mp4` を出力します。

## 主なオプション
- `--fps <float>`: メタデータに FPS が無い場合の代替（既定: 10.0）
- `--codec <fourcc>`: MP4 コーデック（既定: `mp4v`。`avc1` などへ自動フォールバックあり）
- `--overwrite`: 既存の MP4 がある場合に上書き
- `--no-clahe`: CLAHE を無効化（ちらつきや過強調を避けたい場合）
- `--clip-limit <float>`: CLAHE clip limit（既定: 3.0）
- `--tile-grid <int>`: CLAHE タイル数 N（NxN、既定: 8）

### 正規化モード関連
- `--norm-mode {percentile|minmax}`
  - `percentile`（既定）: 外れ値に強い。ムービー全体の画素分布からパーセンタイルで上下を切り、8bitへ線形マッピング
  - `minmax`: 全フレームの最小/最大をそのまま 8bit に線形マッピング
- `--lower-pct <float>`: 下側パーセンタイル（既定: 1.0）
- `--upper-pct <float>`: 上側パーセンタイル（既定: 99.5）

## 実行例
- 既定（パーセンタイル方式）
```bash
python3 nd2_to_mp4.py --input "/Volumes/KINGSTON/20250917PodocyteBis-T"
```
- しきい値を広げて暗部・明部の切り捨てを抑制
```bash
python3 nd2_to_mp4.py --input "/Volumes/KINGSTON/20250917PodocyteBis-T" --lower-pct 0.5 --upper-pct 99.8
```
- 従来の min-max 正規化に戻す
```bash
python3 nd2_to_mp4.py --input "/Volumes/KINGSTON/20250917PodocyteBis-T" --norm-mode minmax --no-clahe
```
- CLAHE を無効化（ちらつき抑制）
```bash
python3 nd2_to_mp4.py --input "/Volumes/KINGSTON/20250917PodocyteBis-T" --no-clahe
```

## 画質について
- 本ツールは最終的に 8bit へ正規化して MP4 を出力します（MP4 は 8bit）。
- ND2 ビューアと見え方が異なる場合は、下記を調整してください。
  - `--no-clahe` で強調無効化
  - `--lower-pct` / `--upper-pct` を広げる、または `--norm-mode minmax` に変更
- 16bit の ND2 でも、読み込み時に配列のエンディアンをネイティブに統一してから処理するため、ビット順の違いによる暗化・コントラスト不正の影響を排除しています。

## パフォーマンス
- パーセンタイル推定は各フレームから画素をサンプリングして行い、コストを抑えています。
- 大容量 ND2 では初回パス（統計計算）がボトルネックになる場合があります。

## トラブルシューティング
- 「一部が暗い / 潰れる」: パーセンタイルを広げる（例: `--lower-pct 0.5 --upper-pct 99.8`）か、`--norm-mode minmax` を使用。
- 「ちらつく」: `--no-clahe` を使用。
- 「書き出せない」: `opencv-python`、`nd2reader` または `pims` がインストールされているか確認。
- 「コーデックでエラー」: `--codec avc1` に変更して再試行。

## MP4 メタデータについて

### 自動埋め込み情報
生成される MP4 ファイルには以下のメタデータが自動的に埋め込まれます：
- **元ファイルパス**: 変換元の ND2 ファイルの絶対パス
- **フィールドサイズ**: 視野の一辺の長さ（マイクロメートル単位）
- **フレームレート**: 実際の FPS 値
- **ファイルタイトル**: MP4 ファイル名

### メタデータの確認方法

#### 1. コマンドライン（最も確実）
```bash
# コメントタグのみ表示
ffprobe -v quiet -show_entries format_tags=comment -of default=noprint_wrappers=1:nokey=1 ファイル名.mp4

# 全メタデータ表示
ffprobe -v quiet -show_entries format_tags -of default=noprint_wrappers=1 ファイル名.mp4
```

#### 2. GUI アプリケーション
- **VLC Media Player**（推奨）: `⌘I` → 「メタデータ」タブ
- **QuickTime Player**: `⌘I` → 「ムービーインスペクタ」
- **MediaInfo**（App Store）: ファイルをドラッグ&ドロップ
- **Subler**（App Store）: MP4 メタデータ専用ツール
- **IINA**: `⌘I` → インスペクタ

#### 3. メタデータ例
```
title: x20x2_BlisterBeads001.mp4
comment: Source: /path/to/file.nd2 | Side: 315.6µm | FPS: 10.000
description: Source: /path/to/file.nd2 | Side: 315.6µm | FPS: 10.000
```

### 注意事項
- macOS の「情報を見る」では MP4 メタデータが表示されない場合があります
- より確実な表示には上記の専用アプリケーションを使用してください
- MOV 形式での出力も可能（拡張子を `.mov` に変更）

## 既知の制限
- 出力は 8bit MP4（可逆ではありません）。完全なビット深度保持が必要な場合は、16bit の TIFF/PNG 連番出力の追加実装が必要です（要望があれば対応可能）。

## バージョン履歴（抜粋）
- 1.2: ムービー全体の min-max 正規化に統合
- 1.3: `--no-clahe` 追加
- 1.4: ヒストグラム（パーセンタイル）方式を既定化、`--norm-mode`/`--lower-pct`/`--upper-pct` 追加
- 1.5: 16bit 等でのエンディアン不一致に対応（ネイティブ化）
- 1.6: MP4 メタデータ自動埋め込み機能追加（元ファイルパス、フィールドサイズ、FPS）

## 参考
- 使用ライブラリ: nd2reader / pims / OpenCV / NumPy
- 例: `python3 nd2_to_mp4.py --help`
