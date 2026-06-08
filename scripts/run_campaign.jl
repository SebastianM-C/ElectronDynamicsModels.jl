#!/usr/bin/env julia
# run_campaign.jl — provision a Hot Aisle MI300X VM, run a set of EDM_* specs on it
# (REPRODUCE one stored run, or expand a base×vary SWEEP), pull back the reduced products
# (PNGs + run manifest — never the raw .jls), and tear the VM down.
#
# Focused by design: rocm backend, a SINGLE git commit per invocation, and the warm-VM
# model — clone + instantiate ONCE, then run every spec on the already-warmed VM.
#
#   julia --project=scripts scripts/run_campaign.jl reproduce path/to/run_<id>.toml
#   julia --project=scripts scripts/run_campaign.jl sweep     path/to/sweep.toml
#
# sweep.toml:
#   commit = "513f94b"                       # must be pushed to the public repo
#   [base]                                   # EDM_* knobs (no prefix) fixed for every run
#   NX = 64
#   NSAMPLES = 2000
#   [vary]                                   # swept knobs → list of values
#   A0 = [1.0e-3, 1.0e-2, 0.1]
#
# Orchestration codifies the steps validated by hand on 2026-06-07; provisioning + the
# IP read-back are the parts to confirm on the first live run.

using RunManifests
using TOML
using JSON

const TEAM = get(ENV, "HOTAISLE_TEAM") do
    f = expanduser("~/.config/hotaisle/team")   # sibling of the token; kept out of the repo
    isfile(f) ? strip(read(f, String)) :
        error("team handle not set — export HOTAISLE_TEAM or write it to $f")
end
const API = "https://admin.hotaisle.app/api/teams/$TEAM"
const TOKEN = strip(read(get(ENV, "HOTAISLE_TOKEN_FILE", expanduser("~/.config/hotaisle/token")), String))
const REPO_URL = "https://github.com/SebastianM-C/ElectronDynamicsModels.jl.git"
const SCRIPT = get(ENV, "EDM_CAMPAIGN_SCRIPT", "lpwa.jl")   # which solver to run on the VM
const LOCAL_OUT = abspath(get(ENV, "EDM_CAMPAIGN_OUT", "campaign_out"))
const MAX_PULL = get(ENV, "EDM_CAMPAIGN_MAX_PULL", "2G")   # pull all derived artifacts; skip only the regenerable raw cube (GB-scale)
const SSH = `/usr/bin/ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20`

# ── Hot Aisle REST via curl; responses parsed with JSON.jl ──
function api(method, path; body = nothing)
    c = `curl -fsS -X $method -H "Authorization: Token $TOKEN"`
    body === nothing || (c = `$c -H "Content-Type: application/json" --data-binary $body`)
    out = read(`$c $API$path`, String)
    return isempty(strip(out)) ? nothing : JSON.parse(out)   # DELETE returns 204/empty
end

balance_usd() = api("GET", "/balance/")["available_balance"] / 100

function provision_mi300x()
    @info "provisioning 1× MI300X…"
    vm = api("POST", "/virtual_machines/"; body = """{"gpus":[{"model":"MI300X","count":1}]}""")
    name, ip = vm["name"], vm["ssh_access"]["ip_address"]
    @info "provisioned" name ip
    return (; name, ip)
end

destroy(name) = (api("DELETE", "/virtual_machines/$name/?force=true"); @info "destroyed $name — billing stopped")

ssh_ok(ip, cmd) = success(`$SSH hotaisle@$ip $cmd`)
ssh_run(ip, cmd) = run(`$SSH hotaisle@$ip $cmd`)

function wait_for_ssh(ip; tries = 40, delay = 15)
    for i in 1:tries
        ssh_ok(ip, "true") && return @info "ssh reachable (~$(i * delay)s)"
        sleep(delay)
    end
    error("VM $ip never became SSH-reachable")
end

"Cold setup, paid once: install Julia, clone the commit, instantiate + precompile (~8 min)."
function warm_vm(ip, commit)
    @info "warming VM: julia + clone @ $commit + instantiate…"
    return ssh_run(
        ip, """
            set -e
            [ -x \$HOME/.juliaup/bin/julia ] || curl -fsSL https://install.julialang.org | sh -s -- --yes
            rm -rf EDM && git clone --quiet $REPO_URL EDM && git -C EDM checkout --quiet $commit
            \$HOME/.juliaup/bin/julia --startup=no --project=EDM/scripts -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
        """
    )
end

"Run one spec on the warm VM, then pull back all derived products (figures, manifest, reduction .jls) — everything except the regenerable raw cube, skipped via --max-size."
function run_one(ip, env)
    env = merge(env, Dict("EDM_GPU_BACKEND" => "rocm"))   # launcher adapts the spec to the target HW
    tag = string(hash(sort(collect(env))); base = 16)
    envline = join(("$k=$v" for (k, v) in env), " ")
    @info "run" tag env
    ssh_run(
        ip, """
            set -e
            cd EDM/scripts
            $envline EDM_OUTDIR=\$HOME/out_$tag \$HOME/.juliaup/bin/julia -t auto --startup=no --project=. $SCRIPT
        """
    )
    dest = joinpath(LOCAL_OUT, "out_$tag")
    mkpath(dest)
    sshe = "/usr/bin/ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
    run(`/usr/bin/rsync -az -e $sshe --max-size=$MAX_PULL hotaisle@$ip:out_$tag/ $dest/`)
    return @info "pulled reductions → $dest"
end

# ── spec sources → (commit, Vector{env}) ─────────────────────────────────────
specs_reproduce(path) = (s = run_spec_from_manifest(TOML.parsefile(path)); (s.commit, [s.env]))

function specs_sweep(path)
    spec = TOML.parsefile(path)
    return (String(spec["commit"]), expand_sweep(get(spec, "base", Dict()), get(spec, "vary", Dict())))
end

function main(args)
    length(args) == 2 || error("usage: run_campaign.jl reproduce|sweep <path>")
    mode, path = args
    commit, envs = mode == "reproduce" ? specs_reproduce(path) :
        mode == "sweep" ? specs_sweep(path) :
        error("unknown mode $(repr(mode)) (use reproduce|sweep)")
    isempty(envs) && error("no runs to execute — is the sweep expansion implemented?")
    @info "campaign start" mode commit runs = length(envs) balance_usd = balance_usd() out = LOCAL_OUT
    vm = provision_mi300x()
    try
        wait_for_ssh(vm.ip)
        warm_vm(vm.ip, commit)
        for (i, env) in enumerate(envs)
            @info "── run $i/$(length(envs)) ──"
            run_one(vm.ip, env)
        end
    finally
        destroy(vm.name)   # always stop the meter — even on error or interrupt
    end
    return @info "campaign done" out = LOCAL_OUT balance_usd = balance_usd()
end

main(ARGS)
