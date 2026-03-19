#!/bin/bash

# RawDataToPresentations インストーラスクリプト
# macOS用の自動セットアップスクリプト

set -e  # エラー時に停止

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ログ関数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ヘルプ表示
show_help() {
    cat << EOF
RawDataToPresentations インストーラ

使用方法:
    $0 [オプション]

オプション:
    -h, --help          このヘルプを表示
    --skip-deps         依存関係のインストールをスキップ
    --skip-symlinks     シンボリックリンクの作成をスキップ
    --force             既存のリンクを上書き
    --dry-run           実際の変更を行わずに実行内容を表示

このスクリプトは以下を実行します:
1. Python依存関係のインストール
2. /Users/\$USER/bin へのシンボリックリンク作成
3. PATH設定の確認

EOF
}

# 設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_BIN_DIR="$HOME/bin"
DRY_RUN=false
SKIP_DEPS=false
SKIP_SYMLINKS=false
FORCE=false

# コマンドライン引数の処理
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --skip-symlinks)
            SKIP_SYMLINKS=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log_error "不明なオプション: $1"
            show_help
            exit 1
            ;;
    esac
done

# ドライラン表示
if [ "$DRY_RUN" = true ]; then
    log_info "ドライランモード: 実際の変更は行いません"
fi

# システム要件チェック
check_requirements() {
    log_info "システム要件をチェック中..."
    
    # macOSチェック
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_error "このスクリプトはmacOS専用です"
        exit 1
    fi
    
    # Python3チェック
    if ! command -v python3 &> /dev/null; then
        log_error "Python3がインストールされていません"
        log_info "Homebrewを使用してインストールしてください: brew install python3"
        exit 1
    fi
    
    # pip3チェック
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3がインストールされていません"
        exit 1
    fi
    
    # Keynoteチェック
    if [ ! -d "/Applications/Keynote.app" ]; then
        log_warning "Keynoteアプリケーションが見つかりません"
        log_warning "App StoreからKeynoteをインストールしてください"
    fi
    
    log_success "システム要件チェック完了"
}

# Python依存関係のインストール
install_dependencies() {
    if [ "$SKIP_DEPS" = true ]; then
        log_info "依存関係のインストールをスキップ"
        return
    fi
    
    log_info "Python依存関係をインストール中..."
    
    # requirements.txtの存在確認
    if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
        log_error "requirements.txtが見つかりません"
        exit 1
    fi
    
    # 基本パッケージのインストール
    local packages=(
        "numpy>=1.21.0"
        "Pillow>=8.0.0"
        "opencv-python"
        "nd2reader"
        "pims"
    )
    
    for package in "${packages[@]}"; do
        log_info "インストール中: $package"
        if [ "$DRY_RUN" = false ]; then
            pip3 install "$package" --user
        else
            echo "  [DRY-RUN] pip3 install $package --user"
        fi
    done
    
    log_success "Python依存関係のインストール完了"
}

# binディレクトリの作成
setup_bin_directory() {
    log_info "binディレクトリをセットアップ中..."
    
    if [ ! -d "$USER_BIN_DIR" ]; then
        log_info "binディレクトリを作成: $USER_BIN_DIR"
        if [ "$DRY_RUN" = false ]; then
            mkdir -p "$USER_BIN_DIR"
        else
            echo "  [DRY-RUN] mkdir -p $USER_BIN_DIR"
        fi
    else
        log_info "binディレクトリは既に存在: $USER_BIN_DIR"
    fi
    
    # PATH設定の確認
    if [[ ":$PATH:" != *":$USER_BIN_DIR:"* ]]; then
        log_warning "PATHに $USER_BIN_DIR が含まれていません"
        log_info "~/.zshrc に以下を追加してください:"
        echo "export PATH=\"\$HOME/bin:\$PATH\""
        
        # 自動追加の提案
        read -p "自動的に~/.zshrcに追加しますか？ (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if [ "$DRY_RUN" = false ]; then
                echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
                log_success "~/.zshrcにPATH設定を追加しました"
                log_info "新しいターミナルセッションで有効になります"
            else
                echo "  [DRY-RUN] echo 'export PATH=\"\$HOME/bin:\$PATH\"' >> ~/.zshrc"
            fi
        fi
    else
        log_success "PATH設定は正しく構成されています"
    fi
}

# シンボリックリンクの作成
create_symlinks() {
    if [ "$SKIP_SYMLINKS" = true ]; then
        log_info "シンボリックリンクの作成をスキップ"
        return
    fi
    
    log_info "シンボリックリンクを作成中..."
    
    # スクリプトファイルのリスト
    local scripts=(
        "nd2movies_to_keynote.py:nd2movies_to_keynote"
        "nd2volumes_to_keynote.py:nd2volumes_to_keynote"
        "nd2FRAP_to_keynote.py:nd2FRAP_to_keynote"
        "nd2FRAP_to_pptx.py:nd2FRAP_to_pptx"
        "nd2_to_mp4.py:nd2_to_mp4"
        "keyenceTIF_to_keynote.py:keyenceTIF_to_keynote"
        "mp4_to_keynote.py:mp4_to_keynote"
        "nd2images_to_keynote.py:nd2images_to_keynote"
    )
    
    for script_info in "${scripts[@]}"; do
        IFS=':' read -r source_file target_name <<< "$script_info"
        source_path="$SCRIPT_DIR/$source_file"
        target_path="$USER_BIN_DIR/$target_name"
        
        # ソースファイルの存在確認
        if [ ! -f "$source_path" ]; then
            log_warning "ソースファイルが見つかりません: $source_file"
            continue
        fi

        # 実行権限が無い場合は付与（新規追加スクリプト対策）
        if [ ! -x "$source_path" ]; then
            if [ "$DRY_RUN" = false ]; then
                chmod +x "$source_path" || true
            else
                echo "  [DRY-RUN] chmod +x \"$source_path\""
            fi
        fi
        
        # 既存のリンクチェック
        if [ -L "$target_path" ] || [ -f "$target_path" ]; then
            if [ "$FORCE" = true ]; then
                log_info "既存のリンクを上書き: $target_name"
                if [ "$DRY_RUN" = false ]; then
                    rm -f "$target_path"
                else
                    echo "  [DRY-RUN] rm -f $target_path"
                fi
            else
                log_warning "既存のリンクをスキップ: $target_name (--force で上書き可能)"
                continue
            fi
        fi
        
        # シンボリックリンク作成
        log_info "リンク作成: $target_name -> $source_file"
        if [ "$DRY_RUN" = false ]; then
            ln -sf "$source_path" "$target_path"
        else
            echo "  [DRY-RUN] ln -sf $source_path $target_path"
        fi
    done
    
    log_success "シンボリックリンクの作成完了"
}

# インストール後の確認
verify_installation() {
    log_info "インストールを確認中..."
    
    local commands=(
        "nd2movies_to_keynote"
        "nd2volumes_to_keynote"
        "nd2FRAP_to_keynote"
        "nd2FRAP_to_pptx"
        "nd2_to_mp4"
        "keyenceTIF_to_keynote"
        "mp4_to_keynote"
        "nd2images_to_keynote"
    )
    
    local all_good=true
    
    for cmd in "${commands[@]}"; do
        if command -v "$cmd" &> /dev/null; then
            log_success "✓ $cmd が利用可能"
        else
            log_error "✗ $cmd が見つかりません"
            all_good=false
        fi
    done
    
    if [ "$all_good" = true ]; then
        log_success "すべてのコマンドが正常にインストールされました！"
    else
        log_error "一部のコマンドのインストールに失敗しました"
        return 1
    fi
}

# 使用例の表示
show_usage_examples() {
    cat << EOF

🎉 インストール完了！

使用例:
    # ND2ムービーからKeynote作成
    nd2movies_to_keynote --input "/path/to/nd2_files"
    
    # ND2ボリューム（3D Zスタック）からKeynote作成
    nd2volumes_to_keynote --input "/path/to/nd2_volumes"

    # FRAP ND2（ROI回復曲線付き）からKeynote作成
    nd2FRAP_to_keynote --input "/path/to/frap_nd2s"

    # FRAP ND2 → PowerPoint（編集可能なチャート・テーブル付き）
    nd2FRAP_to_pptx --input "/path/to/frap_nd2s"
    
    # ND2をMP4に変換
    nd2_to_mp4 --input "/path/to/nd2_files" --fps 10
    
    # Keyence TIFからKeynote作成
    keyenceTIF_to_keynote --input "/path/to/tif_images"
    
    # MP4からKeynote作成
    mp4_to_keynote --input "/path/to/mp4_files"
    
    # ND2画像からKeynote作成
    nd2images_to_keynote --input "/path/to/nd2_images"

詳細な使用方法は README.md を参照してください。

EOF
}

# メイン実行
main() {
    echo "=========================================="
    echo "RawDataToPresentations インストーラ"
    echo "=========================================="
    echo
    
    check_requirements
    install_dependencies
    setup_bin_directory
    create_symlinks
    
    if [ "$DRY_RUN" = false ]; then
        verify_installation
        show_usage_examples
    else
        log_info "ドライラン完了 - 実際のインストールを実行するには --dry-run を外してください"
    fi
}

# スクリプト実行
main "$@"
