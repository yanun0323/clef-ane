#!/bin/sh
# Install the clef-ane command on an Apple Silicon Mac. No Python or Xcode is needed.
#
#   curl -fsSL https://github.com/yanun0323/clef-ane/releases/latest/download/install.sh | sh
#
# Optional settings (environment variables):
#   CLEF_ANE_VERSION      release to install, such as 0.2.0 (default: latest)
#   CLEF_ANE_INSTALL_DIR  program folder (default: ~/.local/share/clef-ane)
#   CLEF_ANE_BIN_DIR      folder for the clef-ane command (default: ~/.local/bin)
#
# Models are downloaded separately with `clef-ane download`.
set -eu

ASSET="clef-ane-macos-arm64.tar.gz"
MARKER="# Created by the clef-ane installer."

say() { printf '%s\n' "$*"; }
die() { printf 'clef-ane install: %s\n' "$*" >&2; exit 1; }

main() {
    repo="${CLEF_ANE_INSTALL_REPO:-https://github.com/yanun0323/clef-ane}"
    version="${CLEF_ANE_VERSION:-latest}"
    install_dir="${CLEF_ANE_INSTALL_DIR:-$HOME/.local/share/clef-ane}"
    bin_dir="${CLEF_ANE_BIN_DIR:-$HOME/.local/bin}"
    if [ -n "${CLEF_ANE_DOWNLOAD_URL:-}" ]; then
        base="$CLEF_ANE_DOWNLOAD_URL"
    elif [ "$version" = latest ]; then
        base="$repo/releases/latest/download"
    else
        base="$repo/releases/download/v${version#v}"
    fi

    [ "$(uname -s)" = Darwin ] || die "clef-ane runs only on macOS."
    [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = 1 ] || die "clef-ane needs a Mac with Apple Silicon (M1 or newer)."
    macos="$(sw_vers -productVersion 2>/dev/null || echo 0)"
    [ "${macos%%.*}" -ge 14 ] 2>/dev/null || die "clef-ane needs macOS 14 or newer (this Mac has $macos)."
    for tool in curl tar shasum mktemp; do
        command -v "$tool" >/dev/null 2>&1 || die "the '$tool' command is missing."
    done

    if [ -e "$bin_dir/clef-ane" ] && ! grep -q "$MARKER" "$bin_dir/clef-ane" 2>/dev/null; then
        die "$bin_dir/clef-ane already exists and was not created by this installer. Remove it, or set CLEF_ANE_BIN_DIR."
    fi

    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    trap 'rm -rf "$tmp"; exit 1' HUP INT TERM

    say "Downloading clef-ane ($version, several hundred MB)..."
    curl -fL --retry 3 --progress-bar -o "$tmp/$ASSET" "$base/$ASSET" || die "download failed: $base/$ASSET"
    curl -fsSL --retry 3 -o "$tmp/$ASSET.sha256" "$base/$ASSET.sha256" || die "checksum download failed."
    expected="$(awk '{print $1; exit}' "$tmp/$ASSET.sha256")"
    actual="$(shasum -a 256 "$tmp/$ASSET" | awk '{print $1}')"
    [ -n "$expected" ] && [ "$expected" = "$actual" ] || die "the download is damaged (checksum mismatch). Please try again."

    say "Installing to $install_dir..."
    parent="$(dirname "$install_dir")"
    mkdir -p "$parent"
    staging="$install_dir.new.$$"
    rm -rf "$staging"
    mkdir "$staging"
    tar -xzf "$tmp/$ASSET" -C "$staging" --strip-components 1 || { rm -rf "$staging"; die "could not unpack the download."; }
    "$staging/bin/clef-ane" --version >/dev/null 2>&1 || { rm -rf "$staging"; die "the downloaded program does not start on this Mac."; }

    if [ -e "$install_dir" ]; then
        mv "$install_dir" "$install_dir.old.$$"
    fi
    mv "$staging" "$install_dir"
    rm -rf "$install_dir.old.$$"

    mkdir -p "$bin_dir"
    cat > "$bin_dir/clef-ane.tmp.$$" <<EOF
#!/bin/sh
$MARKER
exec "$install_dir/bin/clef-ane" "\$@"
EOF
    chmod 755 "$bin_dir/clef-ane.tmp.$$"
    mv "$bin_dir/clef-ane.tmp.$$" "$bin_dir/clef-ane"

    say "Installed $("$bin_dir/clef-ane" --version)."
    case ":$PATH:" in
        *":$bin_dir:"*) ;;
        *)
            say ""
            say "Add $bin_dir to your PATH, then open a new terminal:"
            say "  echo 'export PATH=\"$bin_dir:\$PATH\"' >> ~/.zshrc"
            ;;
    esac
    say ""
    say "Next steps:"
    say "  clef-ane download   # download the model once (about 10 GB)"
    say "  clef-ane serve      # start the local server"
    say ""
    say "To uninstall: rm -rf \"$install_dir\" \"$bin_dir/clef-ane\""
}

main "$@"
