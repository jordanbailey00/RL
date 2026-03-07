#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rl_root="$(cd "$script_dir/.." && pwd)"
workspace_root="${WORKSPACE_ROOT:-$(cd "$rl_root/.." && pwd)}"
tools_dir="$workspace_root/.workspace-tools"
debs_dir="$tools_dir/debs"
sysroot="$tools_dir/sysroot"
bin_dir="$tools_dir/bin"
tmp_dir="$workspace_root/.tmp"

if [ ! -f /etc/os-release ]; then
  echo "Expected /etc/os-release to exist. This bootstrap is intended for Ubuntu WSL." >&2
  exit 1
fi

# PR1 is pinned to the current Ubuntu 24.04 WSL baseline used by the workspace.
. /etc/os-release
if [ "${ID:-}" != "ubuntu" ] || [ "${VERSION_CODENAME:-}" != "noble" ]; then
  echo "This bootstrap is pinned to Ubuntu 24.04 (noble)." >&2
  echo "Detected: ID=${ID:-unknown}, VERSION_CODENAME=${VERSION_CODENAME:-unknown}" >&2
  exit 1
fi

mkdir -p "$debs_dir" "$sysroot" "$bin_dir" "$tmp_dir"

packages=(
  "gcc-13-base=13.2.0-23ubuntu4"
  "cpp-13=13.2.0-23ubuntu4"
  "g++-13=13.2.0-23ubuntu4"
  "gcc-13=13.2.0-23ubuntu4"
  "cpp-13-x86-64-linux-gnu=13.2.0-23ubuntu4"
  "g++-13-x86-64-linux-gnu=13.2.0-23ubuntu4"
  "gcc-13-x86-64-linux-gnu=13.2.0-23ubuntu4"
  "libc6=2.39-0ubuntu8"
  "libc6-dev=2.39-0ubuntu8"
  "libstdc++6=14-20240412-0ubuntu1"
  "libstdc++-13-dev=13.2.0-23ubuntu4"
  "libgcc-s1=14-20240412-0ubuntu1"
  "libgcc-13-dev=13.2.0-23ubuntu4"
  "linux-libc-dev=6.8.0-31.31"
  "libcc1-0=14-20240412-0ubuntu1"
  "libisl23=0.26-3build1"
  "libmpc3=1.3.1-1build1"
)

(
  cd "$debs_dir"
  apt-get download "${packages[@]}"
)

for deb in "$debs_dir"/*.deb; do
  dpkg-deb -x "$deb" "$sysroot"
done

ln -sfn usr/lib "$sysroot/lib"

cat >"$bin_dir/cc" <<'EOF'
#!/usr/bin/env bash

set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sysroot="${WORKSPACE_SYSROOT:-$workspace_root/.workspace-tools/sysroot}"
gcc_bin="$sysroot/usr/bin/x86_64-linux-gnu-gcc-13"

if [ ! -x "$gcc_bin" ]; then
  echo "Missing workspace-local gcc-13 toolchain at $gcc_bin" >&2
  exit 1
fi

exec "$gcc_bin" "--sysroot=$sysroot" "$@"
EOF

cat >"$bin_dir/c++" <<'EOF'
#!/usr/bin/env bash

set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sysroot="${WORKSPACE_SYSROOT:-$workspace_root/.workspace-tools/sysroot}"
gxx_bin="$sysroot/usr/bin/x86_64-linux-gnu-g++-13"

if [ ! -x "$gxx_bin" ]; then
  echo "Missing workspace-local g++-13 toolchain at $gxx_bin" >&2
  exit 1
fi

exec "$gxx_bin" "--sysroot=$sysroot" "$@"
EOF

cat >"$bin_dir/gcc" <<'EOF'
#!/usr/bin/env bash

set -euo pipefail

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cc" "$@"
EOF

cat >"$bin_dir/g++" <<'EOF'
#!/usr/bin/env bash

set -euo pipefail

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/c++" "$@"
EOF

cat >"$bin_dir/ar" <<'EOF'
#!/usr/bin/env bash

set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sysroot="${WORKSPACE_SYSROOT:-$workspace_root/.workspace-tools/sysroot}"
tool="$sysroot/usr/bin/x86_64-linux-gnu-gcc-ar-13"

if [ ! -x "$tool" ]; then
  echo "Missing workspace-local gcc-ar-13 at $tool" >&2
  exit 1
fi

exec "$tool" "$@"
EOF

cat >"$bin_dir/ranlib" <<'EOF'
#!/usr/bin/env bash

set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sysroot="${WORKSPACE_SYSROOT:-$workspace_root/.workspace-tools/sysroot}"
tool="$sysroot/usr/bin/x86_64-linux-gnu-gcc-ranlib-13"

if [ ! -x "$tool" ]; then
  echo "Missing workspace-local gcc-ranlib-13 at $tool" >&2
  exit 1
fi

exec "$tool" "$@"
EOF

chmod +x "$bin_dir/cc" "$bin_dir/c++" "$bin_dir/gcc" "$bin_dir/g++" "$bin_dir/ar" "$bin_dir/ranlib"

echo "Workspace-local GCC sysroot bootstrap complete."
echo "Toolchain root: $sysroot"
echo "Wrapper bin dir: $bin_dir"
