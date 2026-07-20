# campaigns/warm_only.sh — zero-cell campaign: provision + warm + drainer, keep the pod for
# ad-hoc work (e.g. reducing volume-resident cubes). KEEP_CUBE=1 only to trigger start_drainer.
CAMPAIGN=warmup
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
BASE=()
CELLS=()
