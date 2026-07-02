# depot_cache.sh — warm-start the Julia depot on a cloud VM from an rsync-jailed archive store,
# with julia-actions/cache semantics (github.com/julia-actions/cache), which we mirror because a
# VM has no Actions cache service:
#   • archives hold the depot SUBDIRS worth caching: packages artifacts compiled registries
#     scratchspaces (never logs/clones/juliaup — same pick as julia-actions/cache)
#   • the cache key IS the filename: depot-<backend>-<julia version>-<Sys.CPU_NAME>-m<manifest8>.tar.zst
#     CPU name is in the key because pkgimages compile for the host microarch — Hot Aisle bare-metal
#     and RunPod SR-IOV pods only share caches when their CPUs actually match.
#   • restore-keys fallback: exact key → else NEWEST archive on the same prefix (instantiate then
#     tops it up) → else fresh build. Push back ONLY when the exact key was absent, so a Manifest
#     or Julia bump uploads once (from the VM that re-precompiled) and steady-state runs upload nothing.
#
# Sourced ON THE VM (inside a backend's warm heredoc), after julia is on PATH and the repo is cloned.
# Env in:  DEPOT_CACHE          user@host of the jailed store (empty ⇒ every call is a no-op/miss)
#          DEPOT_CACHE_KEYFILE  private key on the VM (default ~/.ssh/depot_key; jailed to the store)
#          BK                   backend slug (rocm|cuda) — namespaces the archives
#          REPO_DIR             repo checkout (for the pinned scripts/Manifest*.toml hash)
#          JULIA_DEPOT_PATH     optional (default ~/.julia)
# Usage:   depot_cache_restore   → sets DC_RESTORED=exact|prefix|miss (+ DC_EXACT/DC_PREFIX)
#          depot_cache_push      → archives the depot under $DC_EXACT (call when DC_RESTORED != exact)
# Degrades gracefully: missing rsync/zstd or an unreachable store ⇒ miss + skipped push (warn only).

_dc_depot() { echo "${JULIA_DEPOT_PATH:-$HOME/.julia}"; }
_dc_ssh()   { echo "ssh -i ${DEPOT_CACHE_KEYFILE:-$HOME/.ssh/depot_key} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes"; }
_dc_ok()    {
    [ -n "${DEPOT_CACHE:-}" ] || return 1
    command -v rsync >/dev/null && command -v zstd >/dev/null && return 0
    echo "[depot-cache] rsync/zstd missing — cache disabled this run" >&2; return 1
}

depot_cache_key() {   # sets DC_PREFIX + DC_EXACT from julia version, CPU microarch, Manifest hash
    local jv cpu mh
    jv=$(julia --version | awk '{print $3}')
    cpu=$(julia --startup=no -e 'print(Sys.CPU_NAME)')
    mh=$(cat "${REPO_DIR:?}"/scripts/Manifest*.toml 2>/dev/null | sha256sum | cut -c1-8)
    DC_PREFIX="depot-${BK:?}-$jv-$cpu"
    DC_EXACT="$DC_PREFIX-m$mh"
}

depot_cache_restore() {
    DC_RESTORED=miss
    _dc_ok || return 0
    depot_cache_key
    # rsync --list-only columns: perms size date time name; keep "date time name" sortable by mtime
    local list best t0
    list=$(rsync --list-only -e "$(_dc_ssh)" "$DEPOT_CACHE:" 2>/dev/null | awk '$1 ~ /^-/ {print $3, $4, $5}') || list=""
    if echo "$list" | grep -q " $DC_EXACT\.tar\.zst\$"; then
        best="$DC_EXACT.tar.zst"; DC_RESTORED=exact
    else
        best=$(echo "$list" | grep " $DC_PREFIX-m[0-9a-f]*\.tar\.zst\$" | sort | tail -1 | awk '{print $3}')
        [ -n "$best" ] && DC_RESTORED=prefix
    fi
    [ -n "$best" ] || { echo "[depot-cache] MISS ($DC_EXACT)"; return 0; }
    t0=$SECONDS
    if rsync --partial -e "$(_dc_ssh)" "$DEPOT_CACHE:$best" /tmp/dc_restore.tar.zst 2>/dev/null; then
        mkdir -p "$(_dc_depot)"
        zstd -dc /tmp/dc_restore.tar.zst | tar -x -C "$(_dc_depot)"
        rm -f /tmp/dc_restore.tar.zst
        echo "[depot-cache] $DC_RESTORED HIT: $best restored in $((SECONDS-t0))s"
    else
        DC_RESTORED=miss; echo "[depot-cache] MISS (download of $best failed)"
    fi
}

depot_cache_push() {
    _dc_ok || return 0
    [ -n "${DC_EXACT:-}" ] || depot_cache_key
    local d dirs=() t0=$SECONDS
    for d in packages artifacts compiled registries scratchspaces; do
        if [ -d "$(_dc_depot)/$d" ]; then dirs+=("$d"); fi
    done
    [ "${#dirs[@]}" -gt 0 ] || { echo "[depot-cache] nothing to push (empty depot?)" >&2; return 0; }
    tar -C "$(_dc_depot)" -cf - "${dirs[@]}" | zstd -T0 -3 > "/tmp/$DC_EXACT.tar.zst"
    if rsync --partial -e "$(_dc_ssh)" "/tmp/$DC_EXACT.tar.zst" "$DEPOT_CACHE:"; then
        echo "[depot-cache] PUSHED $DC_EXACT.tar.zst ($(du -h "/tmp/$DC_EXACT.tar.zst" | cut -f1)) in $((SECONDS-t0))s"
    else
        echo "[depot-cache] push failed — run continues (a future VM rebuilds)" >&2
    fi
    rm -f "/tmp/$DC_EXACT.tar.zst"
}
