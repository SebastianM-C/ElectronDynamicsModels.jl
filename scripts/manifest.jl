# Back-compat shim. The manifest / provenance / reproducibility helpers moved to the
# in-repo package `lib/RunManifests`. Scripts may either `using RunManifests` directly or
# keep `include(joinpath(@__DIR__, "manifest.jl"))` — both bring the same exported names
# (git_state, assert_committed, run_provenance, run_spec_from_manifest, write_derived,
# write_run_manifest, find_parent_manifest, find_parent_run, spp_from_manifest) into scope.
using RunManifests
