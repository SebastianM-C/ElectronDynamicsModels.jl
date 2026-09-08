# ── Static instruction mix ──────────────────────────────────────────────────────────────────
#
# What a kernel's compiled machine code is MADE OF — how many FP64 fused/unfused arithmetic
# instructions, transcendentals, integer ops, memory ops, waits — read straight from the
# vendor disassembly (AMD ISA text, NVIDIA SASS via nvdisasm). Hardware counters give the same
# breakdown dynamically, but only where they work (Nsight Compute; `rocprofv3 --pmc` is not
# available everywhere), and only on hardware one has. A static count is available on every
# target — including one that is not present: the vendor hook can cross-compile the SAME
# kernel (same function, argument types and compile options) for another architecture through
# GPUCompiler, so the CDNA3 / Hopper code can be inspected before renting the machine.
#
# Static ≠ dynamic. The whole-kernel totals count every instruction once, including the cold
# exception paths (bounds checks, DomainErrors) that dominate a Julia kernel's code size but
# never execute. The dynamic figure the totals stand in for is the instruction count of the
# HOT LOOP — for the kernels here, the per-slot loop each thread runs — so the report also
# recovers the loop nest from the control-flow graph (natural loops from dominators) and
# reports the counts inside each loop; `hot_loop` is the outermost loop with the most
# instructions, with a stated confidence. Nested loops are counted ONCE inside their parent,
# so a hot-loop count is a per-iteration floor: a Newton loop nested inside it adds
# (n_iters − 1) × its own count per slot. The LLVM AMDGPU assembly printer annotates blocks
# with its own loop analysis (`; in Loop: Header=BB0_263 Depth=1`); when those annotations are
# present the CFG result is checked against them, which is what backs a `:high` confidence.
#
# The derived figure ([`fp64_issue_floor`](@ref)) combines a per-slot FP64 instruction count
# with the MEASURED FP64 issue rate of the device (`measure_peak_fp64_flops`, an FMA chain at
# 2 FLOP per lane-instruction) into the time the FP64 pipe alone needs per launch — a floor on
# the kernel time on an FP64-issue-bound device, and, against a measured launch time, the
# fraction of the launch the FP64 pipe is provably busy. Every FP64 instruction is assumed to
# issue at the FMA rate (true for DADD/DMUL/`v_add_f64`/`v_mul_f64` on current parts; MUFU/
# `v_rcp_f64` are slower, so the floor errs low).

"""
    MIX_CLASSES

Instruction classes of [`kernel_instruction_mix`](@ref) (vendor-neutral; the manifest keys are
`kernel_mix_<class>`):

- `fp64_fma`, `fp64_add`, `fp64_mul` — FP64 fused multiply-add (DFMA / `v_fma_f64`,
  `v_fmac_f64`, `v_div_fmas_f64`), add/sub, mul;
- `fp64_trans` — FP64 reciprocal / rsqrt / sqrt seeds (MUFU.RCP64H, MUFU.RSQ64H /
  `v_rcp_f64`, `v_rsq_f64`, `v_sqrt_f64`);
- `fp64_other` — every other FP64-pipe instruction: compares (DSETP / `v_cmp_*_f64`),
  min/max, conversions to/from FP64, rounding, the division helpers (`v_div_scale_f64`,
  `v_div_fixup_f64`, `v_ldexp_f64`);
- `fp64_packed` — CDNA packed FP64 (`v_pk_fma_f64`, `v_pk_mul_f64`, `v_pk_add_f64`: two lanes
  of FP64 work per instruction; NVIDIA has no equivalent);
- `fp32` — FP32 and FP16 arithmetic (FFMA/FADD/FMUL/MUFU.RCP/HFMA2 / `v_*_f32`, `v_*_f16`);
- `int` — vector integer / logic / move / select / compare (IMAD, IADD3, LOP3, SHF, ISETP,
  MOV, SEL, PRMT … / `v_add_co_u32`, `v_cndmask_b32`, `v_mov_b32`, `v_readlane_b32`,
  `v_accvgpr_*` …);
- `salu` — scalar / uniform-datapath instructions (AMD `s_*` ALU; NVIDIA `U*` uniform ops,
  R2UR, S2UR);
- `mem_load`, `mem_store`, `mem_atomic` — vector-memory loads, stores, atomics (global, flat,
  local/scratch, buffer);
- `smem` — scalar / constant-bank loads (`s_load_*`; LDC, LDCU, ULDC);
- `lds` — shared memory (`ds_*`; LDS, STS, ATOMS, LDSM);
- `control` — branches, calls, returns, exit, convergence (BSSY/BSYNC/WARPSYNC/BREAK) and
  barriers;
- `wait` — explicit dependency waits and software-scheduled stalls (`s_waitcnt*`, `s_wait_*`,
  RDNA3's `s_delay_alu`; DEPBAR);
- `nop` — no-ops that occupy issue slots (`s_nop`, `v_nop`; NOP — ptxas pads FP64 dependency
  latency with them on the 64:1-rate consumer parts, so they can be a large share of a loop);
- `other` — scheduling hints (`s_clause`, `s_set_inst_prefetch_distance`), cache maintenance
  and fences (`buffer_gl*_inv`, `buffer_wbl2`, MEMBAR, ERRBAR, CCTL), special registers (S2R,
  CS2R), shuffles/votes, and anything unrecognised (see the `opcodes` histogram of the report).
"""
const MIX_CLASSES = (:fp64_fma, :fp64_add, :fp64_mul, :fp64_trans, :fp64_other, :fp64_packed,
    :fp32, :int, :salu, :mem_load, :mem_store, :mem_atomic, :smem, :lds, :control, :wait, :nop, :other)
const FP64_CLASSES = (:fp64_fma, :fp64_add, :fp64_mul, :fp64_trans, :fp64_other, :fp64_packed)

const MixCounts = NamedTuple{MIX_CLASSES, NTuple{length(MIX_CLASSES), Int}}

_zero_counts() = MixCounts(ntuple(_ -> 0, length(MIX_CLASSES)))
function _count_classes(opcodes, classify)
    acc = Dict{Symbol, Int}()
    for op in opcodes
        c = classify(op)
        acc[c] = get(acc, c, 0) + 1
    end
    return MixCounts(ntuple(i -> get(acc, MIX_CLASSES[i], 0), length(MIX_CLASSES)))
end
_add_counts(a::MixCounts, b::MixCounts) = MixCounts(ntuple(i -> a[i] + b[i], length(MIX_CLASSES)))
_fp64_total(c::MixCounts) = sum(c[k] for k in FP64_CLASSES)

# ── Classifiers ─────────────────────────────────────────────────────────────────────────────

# AMD GCN/RDNA/CDNA mnemonics. Encoding suffixes (_e32/_e64/_dpp/_sdwa) are dropped first; the
# RDNA3 dual-issue `v_dual_a :: v_dual_b` line is classified by its first op.
function _classify_amd(op::AbstractString)
    b = replace(String(op), r"_(e32|e64|dpp|dpp8|sdwa)$" => "")
    if startswith(b, "v_")
        b == "v_nop" && return :nop
        occursin(r"^v_pk_\w*_f64$", b) && return :fp64_packed
        if occursin("f64", b)
            occursin(r"^v_(fma|fmac|mad|div_fmas)_f64$", b) && return :fp64_fma
            occursin(r"^v_(add|sub)_f64$", b) && return :fp64_add
            b == "v_mul_f64" && return :fp64_mul
            occursin(r"^v_(rcp|rsq|sqrt)_f64$", b) && return :fp64_trans
            return :fp64_other
        end
        occursin(r"_(f32|f16|bf16)(_|$)", b) && return :fp32
        return :int
    end
    if startswith(b, "s_")
        occursin(r"^s_(buffer_|scratch_)?load", b) && return :smem
        occursin(r"^s_(waitcnt|wait_|delay_alu)", b) && return :wait
        b == "s_nop" && return :nop
        occursin(r"^s_(cbranch|branch|setpc|swappc|call|endpgm|trap|barrier|getpc|rfe)", b) && return :control
        occursin(r"^s_(clause|sleep|sethalt|sendmsg|setprio|inst_prefetch|set_inst_prefetch|code_end|icache|dcache|wakeup|setreg|getreg|ttracedata|endpgm_saved|setkill|singleuse|wait_idle|denorm_mode|round_mode)", b) && return :other
        return :salu
    end
    startswith(b, "ds_") && return :lds
    if occursin(r"^(global|flat|buffer|scratch|tbuffer|image)_", b)
        occursin("atomic", b) && return :mem_atomic
        occursin(r"_(load|sample|gather)", b) && return :mem_load
        occursin("_store", b) && return :mem_store
        return :other      # buffer_gl0_inv, buffer_wbl2, global_wb, …: cache maintenance
    end
    return :other
end

# NVIDIA SASS opcodes (`OPC.MOD1.MOD2` → base + modifiers).
const _SASS_INT = Set(["IMAD", "IADD", "IADD3", "LOP", "LOP3", "SHF", "SHL", "SHR", "SEL", "ISETP",
    "IMNMX", "IABS", "INEG", "LEA", "FLO", "POPC", "BREV", "PRMT", "MOV", "MOV32I", "I2I", "I2IP",
    "VIADD", "VIADDMNMX", "VIMNMX", "VABSDIFF", "VABSDIFF4", "IDP", "IDP4A", "BMSK", "PLOP3", "P2R",
    "R2P", "PSETP", "ISET", "SGXT", "IMUL", "IMUL32I", "XMAD", "ICMP", "BFE", "BFI", "QSPC", "REDUX", "GETLMEMBASE", "SETLMEMBASE", "RPCMOV", "IMADSP", "ISCADD", "ISCADD32I"])
const _SASS_FP32 = Set(["FFMA", "FADD", "FMUL", "FMNMX", "FSETP", "FSEL", "FCHK", "FSWZADD", "FFMA32I",
    "FADD32I", "FMUL32I", "FSET", "FCMP", "HFMA2", "HADD2", "HMUL2", "HSETP2", "HSET2", "HMNMX2", "F2FP",
    "FFMA2", "FADD2", "FMUL2", "HFMA2_32I", "UFSETP", "UFMNMX", "FRND", "F2F", "I2F", "F2I", "I2FP",
    "F2IP", "UI2F", "UF2I", "MUFU"])
const _SASS_SALU = Set(["UMOV", "UIADD3", "ULOP3", "ULOP", "USHF", "USHL", "USHR", "USEL", "UISETP",
    "UIMAD", "ULEA", "UPRMT", "UFLO", "UPOPC", "UPLOP3", "UP2UR", "UR2UP", "R2UR", "S2UR", "UIMNMX",
    "UIADD", "UPSETP", "UBREV", "UBMSK", "USGXT", "VOTEU", "UCGABAR_ARV", "UCGABAR_WAIT", "UISET",
    "UI2I", "UF2FP", "USETMAXREG"])
const _SASS_CONTROL = Set(["BRA", "BRX", "JMP", "JMX", "BRXU", "JMXU", "CALL", "RET", "EXIT", "BSSY",
    "BSYNC", "WARPSYNC", "BREAK", "BPT", "BMOV", "YIELD", "NANOSLEEP", "KILL", "RTT", "PBK", "PCNT",
    "PRET", "BRK", "CONT", "SSY", "SYNC", "BAR", "ACQBULK", "ENDCOLLECTIVE", "PEXIT", "JCAL", "CAL",
    "PLONGJMP", "LONGJMP", "SYNCS", "ELECT", "PMTRIG", "BRXU"])

function _classify_sass(op::AbstractString)
    parts = split(String(op), '.')
    base = parts[1]
    mods = length(parts) > 1 ? join(parts[2:end], ".") : ""
    base == "DFMA" && return :fp64_fma
    base == "DADD" && return :fp64_add
    base == "DMUL" && return :fp64_mul
    if base == "MUFU"
        return occursin("64", mods) ? :fp64_trans : :fp32
    end
    (base == "DSETP" || base == "DMNMX" || base == "DSET") && return :fp64_other
    if base in ("F2F", "I2F", "F2I", "FRND", "I2FP", "F2IP", "UI2F", "UF2I")
        return occursin("F64", mods) ? :fp64_other : :fp32
    end
    base == "DMMA" && return :fp64_other
    base in _SASS_FP32 && return :fp32
    base in _SASS_INT && return :int
    base in _SASS_SALU && return :salu
    base in _SASS_CONTROL && return :control
    if base in ("LDG", "LD", "LDL", "LDGSTS", "LDU", "SULD", "TEX", "TLD", "TLD4", "TXD", "TMML")
        return :mem_load
    end
    base in ("STG", "ST", "STL", "SUST") && return :mem_store
    base in ("ATOM", "ATOMG", "RED", "REDG", "SUATOM", "SURED", "CAS") && return :mem_atomic
    base in ("LDC", "LDCU", "ULDC") && return :smem
    base in ("LDS", "STS", "ATOMS", "LDSM", "STSM", "LDSLK", "STSCUL", "LDSCUL") && return :lds
    base == "DEPBAR" && return :wait
    base == "NOP" && return :nop
    base in ("MEMBAR", "ERRBAR", "CGAERRBAR", "CCTL", "CCTLL", "CCTLT", "FENCE", "S2R", "CS2R",
        "LEPC", "SHFL", "VOTE", "MATCH", "SETCTAID", "LDGDEPBAR", "ARRIVES", "UTMALDG", "UTMASTG",
        "UBLKCP", "UTMACMDFLUSH", "B2R", "R2B", "PIXLD", "VILD", "SETMAXREG", "LEAM") && return :other
    startswith(base, "HMMA") && return :other
    startswith(base, "IMMA") && return :other
    startswith(base, "U") && length(base) > 1 && isuppercase(base[2]) && return :salu   # unknown uniform op
    return :other
end

# ── Disassembly → basic blocks ──────────────────────────────────────────────────────────────

struct MixBlock
    label::String                 # "%bb.18"-style names are unlabelled (fallthrough-only) blocks
    opcodes::Vector{String}
    targets::Vector{String}       # branch target labels (direct branches only)
    fallthrough::Bool             # control may reach the next block in layout order
    loop_note::Union{Nothing, Tuple{String, Int}}   # LLVM asm-printer loop annotation (header label, depth)
end

# Text → blocks for vendor ∈ (:amd, :nvidia). Only the instruction stream, its labels and the
# direct branch targets are kept; directives, debug labels, line info and metadata are skipped.
function _parse_machine_code(text::AbstractString, vendor::Symbol)
    vendor === :amd && return _parse_amd_isa(text)
    vendor === :nvidia && return _parse_sass(text)
    throw(ArgumentError("instruction mix: vendor must be :amd or :nvidia, got $vendor"))
end

const _AMD_TERMINATORS = ("s_branch", "s_endpgm", "s_endpgm_saved", "s_setpc_b64", "s_trap")
const _AMD_UNLABELLED_BLOCK = r"^\s*;\s*%bb\.(\d+):"

function _parse_amd_isa(text::AbstractString)
    blocks = MixBlock[]
    label = nothing            # current block label (nothing before the first instruction)
    opcodes = String[]
    targets = String[]
    note = nothing
    in_metadata = false
    unlabelled = 0
    function finish!()
        (label === nothing && isempty(opcodes)) && return
        lbl = label === nothing ? "%entry" : label
        last_op = isempty(opcodes) ? "" : opcodes[end]
        push!(blocks, MixBlock(lbl, opcodes, targets, !(last_op in _AMD_TERMINATORS), note))
        opcodes = String[]; targets = String[]; note = nothing
        return
    end
    for raw in eachline(IOBuffer(text))
        line = rstrip(raw)
        isempty(line) && continue
        if occursin(r"^\s*\.amdgpu_metadata", line)
            in_metadata = true; continue
        elseif occursin(r"^\s*\.end_amdgpu_metadata", line)
            in_metadata = false; continue
        end
        in_metadata && continue
        m = match(r"^([.\w$]+):", line)
        is_label = m !== nothing && !startswith(line, " ") && !startswith(line, "\t")
        if is_label && occursin(r"^\.L(tmp|func_begin|func_end)\d+$", m[1])
            continue                                   # debug-info labels
        elseif is_label
            finish!()
            label = m[1]
        elseif (m = match(_AMD_UNLABELLED_BLOCK, line)) !== nothing
            finish!()
            label = "%bb." * m[1]
            unlabelled += 1
        end
        # LLVM's loop annotations follow the block marker, on the same line or the next
        m = match(r";\s*in Loop: Header=(\w+) Depth=(\d+)", line)
        if m !== nothing
            note = (".L" * m[1], parse(Int, m[2]))     # "BB0_263" names the label .LBB0_263
            continue
        end
        m = match(r";\s*=>This (?:Inner )?Loop Header: Depth=(\d+)", line)
        if m !== nothing
            note = (something(label, "%entry"), parse(Int, m[1]))
            continue
        end
        (is_label || startswith(something(label, ""), "%bb.") && isempty(opcodes) && occursin(_AMD_UNLABELLED_BLOCK, line)) && continue
        m = match(r"^\s+([a-z][a-z0-9_]*)\b(.*)$", line)
        m === nothing && continue          # directives (\t.loc …), comments, YAML
        op = m[1]
        push!(opcodes, op)
        if startswith(op, "s_cbranch") || op == "s_branch"
            operands = split(m[2], ';')[1]
            t = match(r"(\.L[\w.$]+)", operands)
            t === nothing || push!(targets, t[1])
        end
    end
    finish!()
    return blocks
end

const _SASS_INSTR = r"^\s*/\*[0-9a-f]+\*/\s+(?:(@!?U?P[0-9T])\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;?\s*$"

function _parse_sass(text::AbstractString)
    blocks = MixBlock[]
    label = nothing
    opcodes = String[]
    targets = String[]
    fall = true
    function finish!()
        (label === nothing && isempty(opcodes)) && return
        # code after an unconditional terminator without a label: reachable by nothing
        lbl = label === nothing ? "%bb." * string(length(blocks)) : label
        push!(blocks, MixBlock(lbl, opcodes, targets, fall, nothing))
        opcodes = String[]; targets = String[]; fall = true
        return
    end
    for raw in eachline(IOBuffer(text))
        line = rstrip(raw)
        isempty(line) && continue
        m = match(r"^([.\w$]+):\s*$", line)
        if m !== nothing
            finish!()
            label = m[1]
            continue
        end
        m = match(_SASS_INSTR, line)
        m === nothing && continue
        pred, op, operands = m[1], m[2], m[3]
        push!(opcodes, op)
        base = split(op, '.')[1]
        unconditional = pred === nothing || pred == "@PT" || pred == "@UPT"
        if base in ("BRA", "JMP")
            t = match(r"`\(([^)]+)\)", operands)
            t === nothing || push!(targets, t[1])
            # `BRA.U !UP0, `(.L_x_0)`: a predicate operand makes the branch conditional too
            has_pred_operand = occursin(r"^\s*!?U?P[0-9T]\s*,", operands)
            if unconditional && !has_pred_operand
                fall = false
                finish!(); label = nothing
            end
        elseif base in ("BRX", "JMX", "BRXU", "JMXU", "EXIT", "RET")
            if unconditional
                fall = false
                finish!(); label = nothing
            end
        end
    end
    finish!()
    return blocks
end

# ── Control-flow graph → natural loops ──────────────────────────────────────────────────────

function _cfg_successors(blocks::Vector{MixBlock})
    idx = Dict{String, Int}()
    for (i, b) in enumerate(blocks)
        startswith(b.label, "%") || (idx[b.label] = i)
    end
    succs = [Int[] for _ in blocks]
    for (i, b) in enumerate(blocks)
        for t in b.targets
            j = get(idx, t, 0)
            j == 0 || push!(succs[i], j)
        end
        b.fallthrough && i < length(blocks) && push!(succs[i], i + 1)
        unique!(succs[i])
    end
    return succs
end

# Immediate dominators (Cooper–Harvey–Kennedy) over the graph rooted at a virtual node that
# feeds every block without predecessors (function entries and dead code alike). Returns
# `idom` with `0` for unreachable nodes and `n + 1` for the virtual root.
function _dominators(succs::Vector{Vector{Int}})
    n = length(succs)
    root = n + 1
    preds = [Int[] for _ in 1:root]
    for u in 1:n, v in succs[u]
        push!(preds[v], u)
    end
    roots = [v for v in 1:n if isempty(preds[v])]
    isempty(roots) && n > 0 && push!(roots, 1)
    allsuccs = push!(copy(succs), roots)
    for v in roots
        push!(preds[v], root)
    end
    # reverse postorder from the virtual root
    order = Int[]
    visited = falses(root)
    stack = [(root, 1)]
    visited[root] = true
    while !isempty(stack)
        v, i = stack[end]
        if i <= length(allsuccs[v])
            stack[end] = (v, i + 1)
            w = allsuccs[v][i]
            visited[w] || (visited[w] = true; push!(stack, (w, 1)))
        else
            push!(order, v); pop!(stack)
        end
    end
    reverse!(order)
    rpo = zeros(Int, root)
    for (k, v) in enumerate(order)
        rpo[v] = k
    end
    idom = zeros(Int, root)
    idom[root] = root
    function intersect_(a, b)
        while a != b
            while rpo[a] > rpo[b]
                a = idom[a]
            end
            while rpo[b] > rpo[a]
                b = idom[b]
            end
        end
        return a
    end
    changed = true
    while changed
        changed = false
        for v in order
            v == root && continue
            new = 0
            for p in preds[v]
                idom[p] == 0 && continue
                new = new == 0 ? p : intersect_(p, new)
            end
            if new != 0 && idom[v] != new
                idom[v] = new
                changed = true
            end
        end
    end
    return idom, preds
end

function _dominates(idom, a, b)   # does a dominate b?
    root = length(idom)
    while true
        b == a && return true
        (b == root || b == 0) && return false
        b = idom[b]
    end
end

"""
    _natural_loops(blocks) -> Vector{(header, body, depth)}

Natural loops of the block graph: one per loop header (back edges `u → h` with `h` dominating
`u`, bodies merged per header), with the nesting depth (1 = outermost). Returns block indices.
"""
function _natural_loops(blocks::Vector{MixBlock})
    succs = _cfg_successors(blocks)
    n = length(blocks)
    n == 0 && return NamedTuple{(:header, :body, :depth), Tuple{Int, Vector{Int}, Int}}[]
    idom, preds = _dominators(succs)
    bodies = Dict{Int, Set{Int}}()
    for u in 1:n
        idom[u] == 0 && continue
        for h in succs[u]
            _dominates(idom, h, u) || continue
            body = get!(bodies, h) do
                Set{Int}([h])
            end
            # everything that reaches u without passing through h
            work = [u]
            while !isempty(work)
                x = pop!(work)
                x in body && continue
                push!(body, x)
                for p in preds[x]
                    p <= n && !(p in body) && push!(work, p)
                end
            end
        end
    end
    headers = sort!(collect(keys(bodies)))
    depth = Dict(h => 1 + count(g -> g != h && h in bodies[g], headers) for h in headers)
    loops = [(; header = h, body = sort!(collect(bodies[h])), depth = depth[h]) for h in headers]
    return loops
end

# ── The report ──────────────────────────────────────────────────────────────────────────────

"""
    instruction_mix(text, vendor::Symbol) -> NamedTuple

Static instruction mix of a machine-code listing — the pure part of
[`kernel_instruction_mix`](@ref): `vendor` is `:amd` (LLVM AMDGPU ISA text as `code_native`
prints it) or `:nvidia` (SASS as `nvdisasm --print-code` prints it). Fields:

- `total` — instructions in the listing (every function, every path, each counted once);
- `counts` — a NamedTuple over [`MIX_CLASSES`](@ref); `fp64` its FP64 sum (all six FP64
  classes, packed included);
- `opcodes` — the raw mnemonic histogram (`Dict{String, Int}`), for drilling into `other`;
- `blocks` — basic blocks parsed;
- `loops` — the natural loops of the control-flow graph, largest first, each with `header`
  (label), `depth` (1 = outermost), `blocks`, `total` and `counts` (everything inside the
  loop, nested loops included) and `exclusive_total` / `exclusive_counts` (the loop's own
  blocks only). Trivial loops (≤ 2 instructions of pure control flow, e.g. the trap spin
  after EXIT) are dropped;
- `hot_loop` — the depth-1 loop with the most instructions (`nothing` when there is none):
  for a one-work-item-per-thread kernel with an inner per-slot loop this is that loop, and
  its `counts` are the static per-slot instruction floor (nested loops counted once);
- `hot_loop_confidence` — `:high` when the LLVM assembly printer's own loop annotations are
  present and agree with the CFG analysis (AMD), `:medium` when the CFG has a single
  dominant outer loop (≥ 2× the next), `:low` otherwise, `:none` without loops;
- `llvm_loops_agree` — `true`/`false` when LLVM annotations were present, else `nothing`.
"""
function instruction_mix(text::AbstractString, vendor::Symbol)
    blocks = _parse_machine_code(text, vendor)
    classify = vendor === :amd ? _classify_amd : _classify_sass
    opcodes = Dict{String, Int}()
    for b in blocks, op in b.opcodes
        opcodes[op] = get(opcodes, op, 0) + 1
    end
    block_counts = [_count_classes(b.opcodes, classify) for b in blocks]
    total_counts = reduce(_add_counts, block_counts; init = _zero_counts())
    total = sum(total_counts)

    raw_loops = _natural_loops(blocks)
    bodies = Dict(l.header => Set(l.body) for l in raw_loops)
    loops = map(raw_loops) do l
        nested = [g.header for g in raw_loops if g.header != l.header && g.header in bodies[l.header]]
        excl = setdiff(bodies[l.header], (bodies[g] for g in nested)...)
        c = reduce(_add_counts, (block_counts[i] for i in l.body); init = _zero_counts())
        ce = reduce(_add_counts, (block_counts[i] for i in excl); init = _zero_counts())
        (; header = blocks[l.header].label, depth = l.depth, blocks = length(l.body),
            total = sum(c), counts = c, exclusive_total = sum(ce), exclusive_counts = ce,
            _index = l.header, _body = l.body)
    end
    filter!(l -> !(l.total <= 2 && l.total == l.counts.control + l.counts.other), loops)
    sort!(loops; by = l -> -l.total)

    outer = filter(l -> l.depth == 1, loops)
    hot = isempty(outer) ? nothing : first(outer)
    agree = _llvm_loops_agree(blocks, loops)
    confidence = hot === nothing ? :none :
        agree === true ? :high :
        agree === false ? :low :
        (length(outer) == 1 || outer[1].total >= 2 * outer[2].total) ? :medium : :low

    strip_(l) = (; header = l.header, depth = l.depth, blocks = l.blocks, total = l.total, counts = l.counts,
        exclusive_total = l.exclusive_total, exclusive_counts = l.exclusive_counts)
    return (; vendor, total, fp64 = _fp64_total(total_counts), counts = total_counts, opcodes,
        blocks = length(blocks), loops = map(strip_, loops),
        hot_loop = hot === nothing ? nothing : strip_(hot), hot_loop_confidence = confidence,
        llvm_loops_agree = agree)
end

# LLVM asm-printer loop annotations vs the CFG loops: every annotated block must lie in the
# CFG loop of the header it names, at the same depth, and every annotated header must be a
# CFG header. `nothing` when the listing carries no annotations.
function _llvm_loops_agree(blocks::Vector{MixBlock}, loops)
    any(b -> b.loop_note !== nothing, blocks) || return nothing
    by_header = Dict(l.header => l for l in loops)
    for (i, b) in enumerate(blocks)
        b.loop_note === nothing && continue
        h, d = b.loop_note
        l = get(by_header, h, nothing)
        l === nothing && return false
        (i in l._body && l.depth == d) || return false
    end
    return true
end

"""
    kernel_instruction_mix(backend, ck::CompiledKernel; target = nothing, dump = nothing) -> NamedTuple

Static instruction mix of a compiled kernel (see [`compiled_kernels`](@ref)): the vendor
disassembly of the kernel — the AMD ISA text of `code_native`, the SASS of CUDA.jl's bundled
`nvdisasm` on the cubin — counted by [`MIX_CLASSES`](@ref), with the loop nest recovered from
the control-flow graph and the hot loop's counts. All fields of [`instruction_mix`](@ref),
plus `name`, `signature`, `target` (the ISA the counted code was compiled for: `gfx1100`,
`sm_120a`, …), `native` (whether that is the current device's ISA, i.e. whether the counted
code is the code that ran), and `registers` (the VGPR count read from the AMD listing's
metadata, to cross-check the runtime's attribute; `nothing` for SASS, where `nvdisasm` prints
none — `kernel_resources` has the ptxas figure).

`target = "gfx942"` / `"sm_90"` **cross-compiles** the same kernel — same function, argument
types, `always_inline` and static workgroup size — for another architecture through
GPUCompiler and counts THAT code, without the hardware: what the MI300X / H100 stream looks
like, whether CDNA3 packed FP64 appears, how many waits the scheduler inserted. AMD targets
take the `gfx` name optionally with HIP feature suffixes (`"gfx942:sramecc+:xnack-"`); GCN/CDNA
(`gfx9*`) compile wave64, RDNA (`gfx10+`) wave32, like HIP. NVIDIA targets are `sm_NN`
(`sm_90`, `sm_100`, `sm_120a`, …) as ptxas names them; a CUDA context must exist, not a
device of that architecture. `dump` (an `IO` or a path) receives the disassembly.

Nothing is launched; the compile costs a few seconds. Static counts count code, not
execution: read `hot_loop` for the per-iteration floor of the kernel's main loop and `total`
for the whole binary including its cold exception paths; see [`fp64_issue_floor`](@ref) for
turning the former into time.
"""
function kernel_instruction_mix(backend::KA.Backend, ck::CompiledKernel; target = nothing, dump = nothing)
    code = _kernel_machine_code(backend, ck, target === nothing ? nothing : String(target))
    if dump isa IO
        write(dump, code.text)
    elseif dump !== nothing
        write(String(dump), code.text)
    end
    mix = instruction_mix(code.text, code.vendor)
    return merge((; name = ck.name, signature = ck.signature, target = code.isa, native = code.native,
        registers = code.registers), mix)
end
kernel_instruction_mix(::KA.CPU, ck::CompiledKernel; kwargs...) = error(
    "kernel_instruction_mix: the CPU backend compiles no GPU kernels"
)

# Vendor hook (ext/): the disassembly of `ck` for `target` (nothing = the current device), as
# `(; text, vendor::Symbol, isa::String, native::Bool, registers::Union{Int, Nothing})`.
_kernel_machine_code(b::KA.Backend, ck::CompiledKernel, target) = error(
    "kernel_instruction_mix: no GPU vendor extension loaded for ", typeof(b), " — load CUDA.jl or AMDGPU.jl"
)

"""
    fp64_issue_floor(mix; n_slots, peak_fp64_flops, kernel_time_s = nothing, scope = :hot_loop) -> NamedTuple

The time the FP64 pipe alone needs for a launch, from a STATIC instruction count and a
MEASURED issue rate: `fp64_per_slot` FP64 instructions (the six FP64 classes of `mix`'s
`hot_loop` — `scope = :total` uses the whole-kernel count instead) × `n_slots` executions
(hot-loop iterations summed over all threads; e.g. pixels × samples when every slot is
inside the window) ÷ the device's FP64 lane-instruction rate, taken as
`peak_fp64_flops / 2` (`measure_peak_fp64_flops` runs an FMA chain: 2 FLOP per instruction).
`floor_s` is a lower bound on the launch time on an FP64-issue-bound device;
`fp64_issue_fraction = floor_s / kernel_time_s` (when a measured launch time is given) is
the fraction of the launch during which the FP64 pipe was provably issuing.

Assumptions, stated in `assumptions`: every FP64 instruction issues at the FMA rate (adds
and multiplies do; MUFU/`v_rcp_f64` seeds and packed CDNA ops do not, so the floor errs
low); the static hot-loop count is one pass through the loop with nested loops counted
once — a Newton loop of `n` iterations adds `(n − 1)` × its own count per slot, so the
floor is again low by that much; nothing outside the hot loop (per-thread setup) is counted.
"""
function fp64_issue_floor(mix; n_slots::Real, peak_fp64_flops::Real, kernel_time_s = nothing,
        scope::Symbol = :hot_loop)
    n_slots > 0 || throw(ArgumentError("n_slots must be > 0"))
    peak_fp64_flops > 0 || throw(ArgumentError("peak_fp64_flops must be > 0"))
    counts = if scope === :hot_loop
        mix.hot_loop === nothing && throw(ArgumentError("fp64_issue_floor: the mix has no hot loop (scope = :total counts the whole kernel)"))
        mix.hot_loop.counts
    elseif scope === :total
        mix.counts
    else
        throw(ArgumentError("scope must be :hot_loop or :total, got $scope"))
    end
    fp64 = _fp64_total(counts)
    lane_rate = peak_fp64_flops / 2
    floor_s = fp64 * n_slots / lane_rate
    frac = kernel_time_s === nothing ? NaN : floor_s / kernel_time_s
    return (; scope, fp64_per_slot = fp64, n_slots = Float64(n_slots), fp64_lane_instructions = fp64 * Float64(n_slots),
        peak_fp64_flops = Float64(peak_fp64_flops), floor_s, kernel_time_s, fp64_issue_fraction = frac,
        confidence = scope === :hot_loop ? mix.hot_loop_confidence : :static_total,
        assumptions = "every FP64 instruction issues at the FMA rate; static count = one pass through the loop, nested loops counted once; per-thread setup outside the loop not counted")
end
