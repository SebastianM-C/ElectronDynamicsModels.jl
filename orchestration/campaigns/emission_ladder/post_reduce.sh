# emission_ladder/post_reduce.sh — the ladder's REDUCE_HOOK, sourced by both ladder recipes:
#   REDUCE_HOOK='ladder_reduce "$uuid"'   (run_cell.sh evals it in the cell's background subshell, then
#   deletes the cube if <uuid>.reduced exists — so everything here that reads the cube runs first).
# Under the per-host lock (one cube in RAM per host): the rect reduction (EDM_APODIZATION=none, O_DIRECT
# read — the reducer takes both from its own env, not BASE) and the per-pixel timeseries extraction.
# Then, cube-free: the production checks (shelf_sentinel.jl) on every cell, and on LOCAL numeric cells
# the trajectory diagnostics (pixel traces, worldlines + mass shell + Δγ, window coverage, final
# positions + ic_lg, alias metrics). Those re-solve ODE trajectories, so they do not apply to lpwa.jl's
# analytic ones; cloud lanes run them on the driver after download instead of on billed VM time.
# Set LADDER_SIDE=numeric|lpwa in the recipe.
ladder_reduce() {
    local uuid=$1 m="$CAMP/run_$1.toml" log="$CAMP/run_$1.log" s
    local jl=(nice -n 10 "${JL[@]}" --project=scripts)
    {
        flock 9
        EDM_APODIZATION=none EDM_DIRECT_READ=1 _reduce_cell "$uuid"
        [ -f "$CAMP/$uuid.reduced" ] &&
            ( cd "$REPO" && EDM_DIRECT_READ=1 "${jl[@]}" scripts/extract_screen_timeseries.jl "$m" ) >> "$log" 2>&1
    } 9>"/tmp/edm-reduce-$(id -un).lock"
    [ -f "$CAMP/$uuid.reduced" ] || return 0
    ( cd "$REPO" && "${jl[@]}" scripts/shelf_sentinel.jl "$m" ) >> "$log" 2>&1 ||
        echo "  [ladder] production checks FAILED for $uuid — see derived_shelf_${uuid:0:8}.toml"
    [ "${PROVIDER:-local}" = local ] && [ "${LADDER_SIDE:-}" = numeric ] || return 0
    for s in plot_pixel_traces.jl window_coverage.jl plot_final_positions.jl alias_metrics.jl; do
        ( cd "$REPO" && "${jl[@]}" "scripts/$s" "$m" ) >> "$log" 2>&1 || echo "  [ladder] $s failed for $uuid"
    done
    ( cd "$REPO" && EDM_TRAJ_CHIPS=1 EDM_COORDS=displacement EDM_SOURCE_CAMPAIGN="$CAMP" EDM_SOURCE_RUN="$uuid" \
          "${jl[@]}" scripts/analyze_trajectories.jl ) >> "$log" 2>&1 || echo "  [ladder] worldlines failed for $uuid"
    return 0
}
