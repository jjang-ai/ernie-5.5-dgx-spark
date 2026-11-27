#!/usr/bin/env bash
# Clean-room helper for building, verifying, installing (DKMS + optional MOK signing),
# and uninstalling the 4 KiB sub-page GPU page-table slab allocator for nvidia-uvm
# on 64 KiB ARM64 Linux kernels (e.g., NVIDIA DGX Spark GB10).
#
# License: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/patches/nvidia-580-uvm-subpage-pt.patch"
DKMS_NAME="spark-uvm-subpage-pt"

usage() {
    cat <<'EOF'
Usage:
  spark_uvm_64k_slab.sh status
  spark_uvm_64k_slab.sh build [--kernel <kver>] [--src <nvidia-src-dir>] [--out <dir>] [--force-64k-check]
  spark_uvm_64k_slab.sh install-dkms [--kernel <kver>] [--src <nvidia-src-dir>] [--sign-key <key.priv>] [--sign-cert <cert.der>]
  spark_uvm_64k_slab.sh uninstall [--kernel <kver>]

Commands:
  status        Read-only check of kernel, PAGE_SIZE, MemTotal, Secure Boot,
                installed nvidia-580 source, and live nvidia_uvm sub-page stats.
  build         Copy /usr/src/nvidia-580.* into a staging dir, apply patch cleanly,
                and compile nvidia-uvm.ko against target kernel headers without
                modifying any installed system module.
  install-dkms  Register a DKMS override module (spark-uvm-subpage-pt) that builds,
                optionally signs with a MOK keypair, and installs nvidia-uvm.ko into
                /lib/modules/<kver>/updates/dkms/.
  uninstall     Remove the DKMS override and run depmod to restore stock nvidia-uvm.ko.
EOF
}

detect_nvidia_src() {
    local cand
    for cand in /usr/src/nvidia-580.*; do
        if [ -d "$cand/nvidia-uvm" ]; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

cmd_status() {
    local kver page_sz sb_state src_dir mem_total_kb
    kver="$(uname -r)"
    page_sz="$(getconf PAGESIZE 2>/dev/null || echo unknown)"
    mem_total_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
    sb_state="unknown"
    if command -v mokutil >/dev/null 2>&1; then
        sb_state="$(mokutil --sb-state 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
    fi
    src_dir="$(detect_nvidia_src 2>/dev/null || echo 'not-found')"

    echo "=== DGX Spark 64 KiB UVM Page-Table Slab Status ==="
    echo "Kernel:           $kver"
    echo "Host PAGE_SIZE:   ${page_sz} bytes"
    echo "MemTotal:         $((mem_total_kb / 1024)) MiB ($mem_total_kb KiB)"
    echo "Secure Boot:      $sb_state"
    echo "NVIDIA UVM src:   $src_dir"
    if [ -r /sys/module/nvidia_uvm/version ]; then
        echo "Loaded UVM ver:   $(cat /sys/module/nvidia_uvm/version)"
    else
        echo "Loaded UVM ver:   (module not loaded)"
    fi
    if [ -r /sys/module/nvidia_uvm/parameters/uvm_pt_subpage_stats ]; then
        echo "Sub-page slab:    ACTIVE"
        echo "Live telemetry:   $(cat /sys/module/nvidia_uvm/parameters/uvm_pt_subpage_stats)"
    else
        echo "Sub-page slab:    NOT LOADED (stock nvidia_uvm or 4 KiB kernel)"
    fi
    if command -v dkms >/dev/null 2>&1; then
        echo "DKMS status:"
        dkms status "$DKMS_NAME" 2>/dev/null || echo "  (not installed)"
    fi
}

cmd_build() {
    local kver="$(uname -r)"
    local src_dir=""
    local out_dir="/tmp/spark-uvm-subpage-build"
    local force_64k_check=0

    while [ $# -gt 0 ]; do
        case "$1" in
            --kernel) kver="$2"; shift 2 ;;
            --src) src_dir="$2"; shift 2 ;;
            --out) out_dir="$2"; shift 2 ;;
            --force-64k-check) force_64k_check=1; shift ;;
            *) echo "Unknown arg: $1" >&2; exit 2 ;;
        esac
    done

    if [ -z "$src_dir" ]; then
        src_dir="$(detect_nvidia_src)" || {
            echo "ERROR: Could not find /usr/src/nvidia-580.* open driver source." >&2
            exit 1
        }
    fi

    if [ ! -d "/lib/modules/$kver/build" ]; then
        echo "ERROR: Kernel headers missing at /lib/modules/$kver/build" >&2
        exit 1
    fi

    echo "[1/4] Staging $src_dir -> $out_dir (read-only source preserved)..."
    rm -rf "$out_dir"
    mkdir -p "$out_dir"
    cp -a "$src_dir/." "$out_dir/"

    echo "[2/4] Dry-run & applying clean-room patch..."
    patch -d "$out_dir" -p1 --dry-run < "$PATCH_FILE"
    patch -d "$out_dir" -p1 < "$PATCH_FILE"

    if [ "$force_64k_check" -eq 1 ]; then
        echo "[2b] Compile-check mode: overriding #if PAGE_SIZE > 4096 guard locally to syntax-check 64k slab code on 4k headers..."
        python3 - "$out_dir/nvidia-uvm/uvm_mmu.h" "$out_dir/nvidia-uvm/uvm_mmu.c" <<'PYEOF'
import sys
from pathlib import Path
for path_str in sys.argv[1:]:
    p = Path(path_str)
    s = p.read_text()
    s = s.replace("#if PAGE_SIZE > 4096", "#if 1")
    s = s.replace("#define UVM_PT_SUBPAGE_SLOTS (PAGE_SIZE / UVM_PT_SUBPAGE_SIZE)", "#define UVM_PT_SUBPAGE_SLOTS 16U")
    p.write_text(s)
PYEOF
    fi

    echo "[3/4] Building nvidia-uvm.ko for kernel $kver..."
    make -C "$out_dir" \
        KERNEL_UNAME="$kver" \
        NV_KERNEL_MODULES="nvidia nvidia-uvm" \
        -j"$(nproc)" \
        modules

    echo "[4/4] Built module:"
    ls -lh "$out_dir/nvidia-uvm.ko"
    modinfo "$out_dir/nvidia-uvm.ko" | grep -E '^(filename|vermagic|parm:.*uvm_pt_subpage)' || true
}

cmd_install_dkms() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "ERROR: install-dkms must be run as root (sudo)." >&2
        exit 1
    fi
    command -v dkms >/dev/null 2>&1 || {
        echo "ERROR: dkms is not installed. Run: sudo apt-get install -y dkms" >&2
        exit 1
    }

    local kver="$(uname -r)"
    local src_dir=""
    local sign_key="/var/lib/shim-signed/mok/MOK.priv"
    local sign_cert="/var/lib/shim-signed/mok/MOK.der"

    while [ $# -gt 0 ]; do
        case "$1" in
            --kernel) kver="$2"; shift 2 ;;
            --src) src_dir="$2"; shift 2 ;;
            --sign-key) sign_key="$2"; shift 2 ;;
            --sign-cert) sign_cert="$2"; shift 2 ;;
            *) echo "Unknown arg: $1" >&2; exit 2 ;;
        esac
    done

    if [ -z "$src_dir" ]; then
        src_dir="$(detect_nvidia_src)" || {
            echo "ERROR: Could not find /usr/src/nvidia-580.* open driver source." >&2
            exit 1
        }
    fi

    local drv_ver
    drv_ver="$(basename "$src_dir" | sed 's/^nvidia-//')"
    local dkms_src="/usr/src/${DKMS_NAME}-${drv_ver}"

    echo "[1/4] Preparing DKMS source tree $dkms_src (driver $drv_ver)..."
    dkms remove -m "$DKMS_NAME" -v "$drv_ver" --all 2>/dev/null || true
    rm -rf "$dkms_src"
    mkdir -p "$dkms_src"
    cp -a "$src_dir/." "$dkms_src/"
    patch -d "$dkms_src" -p1 < "$PATCH_FILE"

    cat > "$dkms_src/dkms.conf" <<EOF
PACKAGE_NAME="${DKMS_NAME}"
PACKAGE_VERSION="${drv_ver}"
AUTOINSTALL="yes"
MAKE[0]="make KERNEL_UNAME=\${kernelver} NV_KERNEL_MODULES='nvidia nvidia-uvm' -j\$(nproc) modules"
CLEAN="make clean"
BUILT_MODULE_NAME[0]="nvidia-uvm"
DEST_MODULE_LOCATION[0]="/updates/dkms"
EOF

    echo "[2/4] Adding, building, and installing DKMS module for $kver..."
    dkms add -m "$DKMS_NAME" -v "$drv_ver"
    dkms build -m "$DKMS_NAME" -v "$drv_ver" -k "$kver"
    dkms install -m "$DKMS_NAME" -v "$drv_ver" -k "$kver" --force

    local mod_path
    mod_path="$(find "/lib/modules/$kver/updates/dkms" -name 'nvidia-uvm.ko*' | head -n 1 || true)"
    if [ -n "$mod_path" ] && [ -f "$sign_key" ] && [ -f "$sign_cert" ]; then
        local sign_file="/lib/modules/$kver/build/scripts/sign-file"
        if [ -x "$sign_file" ]; then
            echo "[3/4] Signing $mod_path with MOK keypair..."
            case "$mod_path" in
                *.zst)
                    zstd -d -f "$mod_path" -o "${mod_path%.zst}"
                    "$sign_file" sha256 "$sign_key" "$sign_cert" "${mod_path%.zst}"
                    zstd -f --rm "${mod_path%.zst}" -o "$mod_path"
                    ;;
                *.xz)
                    xz -d -f "$mod_path"
                    "$sign_file" sha256 "$sign_key" "$sign_cert" "${mod_path%.xz}"
                    xz -f "${mod_path%.xz}"
                    ;;
                *.ko)
                    "$sign_file" sha256 "$sign_key" "$sign_cert" "$mod_path"
                    ;;
            esac
        fi
    else
        echo "[3/4] Skipping MOK signing (key/cert not present at $sign_key / $sign_cert)."
        echo "      If Secure Boot is enabled, enroll a MOK key with mokutil and re-run with --sign-key/--sign-cert."
    fi

    echo "[4/4] Updating module dependencies..."
    depmod -a "$kver"
    echo "Done. Reboot into $kver and run: $0 status"
}

cmd_uninstall() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "ERROR: uninstall must be run as root (sudo)." >&2
        exit 1
    fi
    local kver="$(uname -r)"
    while [ $# -gt 0 ]; do
        case "$1" in
            --kernel) kver="$2"; shift 2 ;;
            *) echo "Unknown arg: $1" >&2; exit 2 ;;
        esac
    done

    if command -v dkms >/dev/null 2>&1; then
        local ver
        for ver in $(dkms status "$DKMS_NAME" 2>/dev/null | awk -F'[,/: ]+' '{print $2}' | sort -u); do
            dkms remove -m "$DKMS_NAME" -v "$ver" --all || true
            rm -rf "/usr/src/${DKMS_NAME}-${ver}"
        done
    fi
    rm -f "/lib/modules/$kver/updates/dkms/nvidia-uvm.ko"*
    depmod -a "$kver"
    echo "Removed $DKMS_NAME and restored stock nvidia-uvm module resolution for $kver."
}

case "${1:-}" in
    status) shift; cmd_status "$@" ;;
    build) shift; cmd_build "$@" ;;
    install-dkms) shift; cmd_install_dkms "$@" ;;
    uninstall) shift; cmd_uninstall "$@" ;;
    -h|--help|"") usage ;;
    *) echo "Unknown command: $1" >&2; usage; exit 2 ;;
esac
