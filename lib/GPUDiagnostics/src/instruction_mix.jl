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
# Counts per class; `:unclassified` (the catch-all rule) lands in `other` and, when a
# histogram Dict is passed, is also recorded there by mnemonic.
function _count_classes(opcodes, vendor::Symbol, unclassified = nothing)
    acc = Dict{Symbol, Int}()
    for op in opcodes
        c = _classify(op, vendor)
        if c === :unclassified
            c = :other
            unclassified === nothing || (unclassified[op] = get(unclassified, op, 0) + 1)
        end
        acc[c] = get(acc, c, 0) + 1
    end
    return MixCounts(ntuple(i -> get(acc, MIX_CLASSES[i], 0), length(MIX_CLASSES)))
end
_add_counts(a::MixCounts, b::MixCounts) = MixCounts(ntuple(i -> a[i] + b[i], length(MIX_CLASSES)))
_fp64_total(c::MixCounts) = sum(c[k] for k in FP64_CLASSES)

# ── Classifiers: ordered rule tables ────────────────────────────────────────────────────────
#
# One generic matcher, two vendor tables, first matching rule wins. The rules follow the
# mnemonic grammars — SASS: a leading operand-type letter (`D` fp64, `F`/`H` fp32/half, `I`
# integer, `U` uniform datapath), `LD`/`ST`/`ATOM`/`RED` + a space suffix (`G` global, `L`
# local, `S` shared, `C` constant) for memory; AMD: `v_`/`s_`/`ds_`/`global_`… prefixes and the
# `_f64`/`_f32` type suffixes — with a SHORT exception list ahead of each generic rule where
# the grammar lies (FLO is integer, FENCE is a fence, MUFU.RCP64H is FP64, F2I.….F64 converts
# FP64, ULDC is a constant load, UTMA* are TMA ops). The last rule is the catch-all
# `:unclassified`, counted as `other` and reported in `unclassified_opcodes` / `coverage`.
# Deliberate `:other` rules (cache maintenance, fences, special registers, hints) are NOT
# unclassified. Users may `pushfirst!(SASS_RULES, r"^FOO" => :int)` to override.

"""
    SASS_RULES :: Vector{Pair{Regex, Symbol}}

Ordered classification rules of [`instruction_mix`](@ref) for NVIDIA SASS mnemonics (matched
against the full `OPC.MOD…` opcode) and AMD ISA mnemonics (matched after the `_e32`/`_e64`/
`_dpp`/`_sdwa` encoding suffix is dropped). The first matching rule wins; the final `r""` rule
is the catch-all `:unclassified` (counted as `other`, listed in `unclassified_opcodes`). The
vectors are mutable: `pushfirst!(SASS_RULES, r"^MYOP" => :int)` overrides for a session.
"""
const SASS_RULES = Pair{Regex, Symbol}[
    # grammar exceptions first
    r"^NOP$" => :nop,
    r"^DEPBAR" => :wait,
    r"^MUFU\.\w*64" => :fp64_trans,                       # MUFU.RCP64H / RSQ64H
    r"^MUFU" => :fp32,
    r"^U?(F2F|F2I|I2F|FRND|I2FP|F2IP)\..*F64" => :fp64_other,   # conversions naming F64 in a modifier
    r"^U?(F2F|F2I|I2F|FRND|I2FP|F2IP|F2FP)(\.|$)" => :fp32,
    r"^FLO(\.|$)" => :int,                                # find leading one
    r"^(MEMBAR|ERRBAR|CGAERRBAR|CCTL|CCTLL|CCTLT|FENCE|S2R|CS2R|LEPC|SHFL|VOTE|VOTEU|MATCH|SETCTAID|LDGDEPBAR|ARRIVES|UTMALDG|UTMASTG|UBLKCP|UTMACMDFLUSH|B2R|R2B|PIXLD|VILD|SETMAXREG|USETMAXREG|LEAM|HMMA|IMMA|BMMA|DMMA)(\.|$)" => :other,
    r"^(BRA|BRX|JMP|JMX|BRXU|JMXU|CALL|RET|EXIT|BSSY|BSYNC|WARPSYNC|BREAK|BPT|BMOV|YIELD|NANOSLEEP|KILL|RTT|PBK|PCNT|PRET|BRK|CONT|SSY|SYNC|BAR|ACQBULK|ENDCOLLECTIVE|PEXIT|JCAL|CAL|PLONGJMP|LONGJMP|SYNCS|ELECT|PMTRIG)(\.|$)" => :control,
    # memory: LD/ST/ATOM/RED + space suffix
    r"^(LDS|STS|LDSM|STSM|ATOMS|LDSLK|STSCUL|LDSCUL)(\.|$)" => :lds,
    r"^U?LDCU?(\.|$)" => :smem,
    r"^(LD|LDG|LDL|LDU|LDGSTS|SULD|TEX|TLD|TLD4|TXD|TMML)(\.|$)" => :mem_load,
    r"^(ST|STG|STL|SUST)(\.|$)" => :mem_store,
    r"^(ATOM|ATOMG|RED|REDG|CAS|SUATOM|SURED)(\.|$)" => :mem_atomic,
    # datapaths by leading operand-type letter
    r"^D" => :fp64_other,                                 # DSETP, DMNMX, DSET, … (DFMA/DADD/DMUL below)
    r"^(R2UR|S2UR|UR2UP|UP2UR)(\.|$)" => :salu,
    r"^U[A-Z]" => :salu,                                  # uniform datapath: UMOV, UIADD3, ULOP3, USHF, USEL, UISETP, …
    r"^[FH]" => :fp32,                                    # FFMA, FADD, FMUL, FMNMX, FSETP, FSEL, HFMA2, HADD2, …
    r"^(I|LOP|SH[FLR]|LEA|PRMT|SEL|MOV|P|BREV|POPC|BMSK|SGXT|VI|VABSDIFF|XMAD|BF[EI]|QSPC|REDUX|GETLMEMBASE|SETLMEMBASE|RPCMOV|R2P)" => :int,
    r"" => :unclassified,
]
# DFMA/DADD/DMUL ahead of the generic `D` rule
pushfirst!(SASS_RULES, r"^DFMA(\.|$)" => :fp64_fma, r"^DADD(\.|$)" => :fp64_add, r"^DMUL(\.|$)" => :fp64_mul)

"""
    AMD_RULES :: Vector{Pair{Regex, Symbol}}

Ordered classification rules of [`instruction_mix`](@ref) for AMD ISA mnemonics (GCN / RDNA /
CDNA), matched after the `_e32`/`_e64`/`_dpp`/`_sdwa` encoding suffix is dropped. Same contract
as [`SASS_RULES`](@ref): first match wins, the final rule is the `:unclassified` catch-all, and
the vector is mutable for session overrides.
"""
const AMD_RULES = Pair{Regex, Symbol}[
    r"^[vs]_nop$" => :nop,
    r"^v_pk_\w*_f64$" => :fp64_packed,
    r"^v_(fma|fmac|mad|div_fmas)_f64$" => :fp64_fma,
    r"^v_(add|sub)_f64$" => :fp64_add,
    r"^v_mul_f64$" => :fp64_mul,
    r"^v_(rcp|rsq|sqrt)_f64$" => :fp64_trans,
    r"^v_\w*f64" => :fp64_other,                          # v_cmp_*_f64, v_cvt_*, v_div_scale/fixup, v_ldexp, v_max, …
    r"^v_\w*_(f32|f16|bf16)(_|$)" => :fp32,
    r"^v_" => :int,                                       # VALU integer / logic / move / select / lane ops
    r"^s_(buffer_|scratch_)?load" => :smem,
    r"^s_(waitcnt|wait_|delay_alu)" => :wait,
    r"^s_(cbranch|branch|setpc|swappc|call|endpgm|trap|barrier|getpc|rfe)" => :control,
    r"^s_(clause|sleep|sethalt|sendmsg|setprio|inst_prefetch|set_inst_prefetch|code_end|icache|dcache|wakeup|setreg|getreg|ttracedata|setkill|singleuse|wait_idle|denorm_mode|round_mode)" => :other,
    r"^s_" => :salu,
    r"^ds_" => :lds,
    r"^(global|flat|buffer|scratch|tbuffer|image)_\w*atomic" => :mem_atomic,
    r"^(global|flat|buffer|scratch|tbuffer|image)_\w*(load|sample|gather)" => :mem_load,
    r"^(global|flat|buffer|scratch|tbuffer|image)_\w*store" => :mem_store,
    r"^(buffer_gl[01]_inv|buffer_inv|buffer_wb\w*|global_(wb|inv))$" => :other,   # cache maintenance
    r"" => :unclassified,
]

# Vendor normalisation before matching: AMD encoding suffixes carry no class information.
_normalize_opcode(op::AbstractString, ::Val{:amd}) = replace(String(op), r"_(e32|e64|dpp|dpp8|sdwa)$" => "")
_normalize_opcode(op::AbstractString, ::Val{:nvidia}) = String(op)
_rules(::Val{:amd}) = AMD_RULES
_rules(::Val{:nvidia}) = SASS_RULES

function _classify(op::AbstractString, rules::Vector{Pair{Regex, Symbol}})
    for (re, cls) in rules
        occursin(re, op) && return cls
    end
    return :unclassified
end
_classify(op::AbstractString, vendor::Symbol) = _classify(_normalize_opcode(op, Val(vendor)), _rules(Val(vendor)))

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
- `unclassified`, `unclassified_opcodes`, `coverage` — instructions no rule of
  [`SASS_RULES`](@ref) / [`AMD_RULES`](@ref) but the catch-all matched (they count as `other`),
  their histogram, and `1 − unclassified / total`. Deliberate `other` rules (fences, cache
  maintenance, special registers, hints) are classified, not unclassified;
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
    opcodes = Dict{String, Int}()
    for b in blocks, op in b.opcodes
        opcodes[op] = get(opcodes, op, 0) + 1
    end
    unclassified_opcodes = Dict{String, Int}()
    block_counts = [_count_classes(b.opcodes, vendor, unclassified_opcodes) for b in blocks]
    total_counts = reduce(_add_counts, block_counts; init = _zero_counts())
    total = sum(total_counts)
    unclassified = sum(values(unclassified_opcodes); init = 0)

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
        unclassified, unclassified_opcodes, coverage = total == 0 ? 1.0 : 1 - unclassified / total,
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

`ir = true` (default) adds the `ir` field: [`kernel_ir_mix`](@ref) of the same job — the
typed LLVM-IR operation counts BEFORE the backend, next to the machine-code counts (an IR
`fp64_fma` of 0 against 219 DFMA per slot is the backend's FMA contraction).

Nothing is launched; the compile costs a few seconds. Static counts count code, not
execution: read `hot_loop` for the per-iteration floor of the kernel's main loop and `total`
for the whole binary including its cold exception paths; see [`fp64_issue_floor`](@ref) for
turning the former into time.
"""
function kernel_instruction_mix(backend::KA.Backend, ck::CompiledKernel; target = nothing, dump = nothing,
        ir::Bool = true)
    tgt = target === nothing ? nothing : String(target)
    code = _kernel_machine_code(backend, ck, tgt)
    if dump isa IO
        write(dump, code.text)
    elseif dump !== nothing
        write(String(dump), code.text)
    end
    mix = instruction_mix(code.text, code.vendor)
    irc = ir ? kernel_ir_mix(backend, ck; target = tgt) : nothing
    return merge((; name = ck.name, signature = ck.signature, target = code.isa, native = code.native,
        registers = code.registers), mix, (; ir = irc))
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

# ── Typed LLVM-IR operation count (vendor-neutral cross-check) ──────────────────────────────
#
# The optimized LLVM module of the same CompilerJob, walked with LLVM.jl and counted by opcode
# and operand TYPE — no text matching. This is the arithmetic as the front end and the
# middle end left it: `fmul`/`fadd` pairs still separate (Julia emits no `fma` for `a*b+c`),
# `fdiv` and `llvm.sqrt.f64` still single operations. The machine code differs from it by
# exactly the backend's work: FMA contraction (IR fma 0 → DFMA / `v_fma_f64`), division and
# square-root expansion into seeds + FMA sequences, CSE and rematerialisation. The walker takes
# the LLVM.jl module binding as an argument (the extensions pass their `LLVM`, a weak
# dependency of this package) so one implementation serves both vendors and the CPU tests.

"""
    IR_CLASSES

Operation classes of [`kernel_ir_mix`](@ref): `fp64_fma` (`llvm.fma.f64`, `llvm.fmuladd.f64`),
`fp64_add` (`fadd`/`fsub double`), `fp64_mul`, `fp64_div`, `fp64_neg`, `fp64_sqrt`
(`llvm.sqrt.f64`), `fp64_cmp` (`fcmp` on doubles), `fp64_cvt` (conversions to/from double),
`fp64_intrinsic` (other `llvm.*` intrinsics returning or taking double: fabs, floor, minnum,
…), `fp64_contract` (the subset of the double `fadd`/`fsub`/`fmul` that carry LLVM's `contract`
fast-math flag — Julia lowers `muladd` to a contract-flagged pair rather than an `fmuladd`
intrinsic, and the backend fuses them at instruction selection; informational, not added to
the `fp64` total), `fp32` (the same on float/half), `int` (integer arithmetic, logic, shifts, compares,
integer casts, selects), `mem_load`, `mem_store`, `mem_atomic`, `call` (non-intrinsic calls),
`control` (br, switch, ret, unreachable), `other` (phi, getelementptr, allocas, pointer casts,
extract/insertvalue, …).
"""
const IR_CLASSES = (:fp64_fma, :fp64_add, :fp64_mul, :fp64_div, :fp64_neg, :fp64_sqrt, :fp64_cmp, :fp64_cvt,
    :fp64_intrinsic, :fp64_contract, :fp32, :int, :mem_load, :mem_store, :mem_atomic, :call, :control, :other)
const IR_FP64_CLASSES = (:fp64_fma, :fp64_add, :fp64_mul, :fp64_div, :fp64_neg, :fp64_sqrt, :fp64_cmp, :fp64_cvt, :fp64_intrinsic)
const IRCounts = NamedTuple{IR_CLASSES, NTuple{length(IR_CLASSES), Int}}

# Walk an LLVM module (`L` = the LLVM.jl module binding) → per-function IRCounts.
function _ir_counts(L::Module, mod)
    API = L.API
    # LLVM.jl names the type structs LLVMDouble/LLVMFloat/LLVMHalf/LLVMBFloat (DoubleType() etc. are constructors)
    isdouble(t) = t isa L.LLVMDouble || (t isa L.VectorType && eltype(t) isa L.LLVMDouble)
    isfp(t) = t isa L.LLVMFloat || t isa L.LLVMHalf || t isa L.LLVMBFloat || t isa L.LLVMDouble ||
        (t isa L.VectorType && isfp(eltype(t)))
    first_operand_type(inst) = L.value_type(first(L.operands(inst)))
    fp_arith = Dict(API.LLVMFAdd => :add, API.LLVMFSub => :add, API.LLVMFMul => :mul, API.LLVMFDiv => :div,
        API.LLVMFRem => :intrinsic, API.LLVMFNeg => :neg)
    int_ops = (API.LLVMAdd, API.LLVMSub, API.LLVMMul, API.LLVMUDiv, API.LLVMSDiv, API.LLVMURem, API.LLVMSRem,
        API.LLVMShl, API.LLVMLShr, API.LLVMAShr, API.LLVMAnd, API.LLVMOr, API.LLVMXor, API.LLVMICmp,
        API.LLVMTrunc, API.LLVMZExt, API.LLVMSExt, API.LLVMSelect)
    cvt_ops = (API.LLVMSIToFP, API.LLVMUIToFP, API.LLVMFPToSI, API.LLVMFPToUI, API.LLVMFPExt, API.LLVMFPTrunc)
    ctl_ops = (API.LLVMBr, API.LLVMSwitch, API.LLVMRet, API.LLVMUnreachable, API.LLVMIndirectBr, API.LLVMInvoke)
    fp64_or_32(kind, dbl) = dbl ? Symbol(:fp64_, kind) : :fp32
    # LLVM's `contract` fast-math flag on a double fadd/fsub/fmul (what `muladd` lowers to)
    contract_flag(inst) = isdefined(L, :fast_math) && isdefined(API, :LLVMCanValueUseFastMathFlags) &&
        Bool(API.LLVMCanValueUseFastMathFlags(inst)) && L.fast_math(inst).contract
    contractible = (API.LLVMFAdd, API.LLVMFSub, API.LLVMFMul)
    per_function = Dict{String, IRCounts}()
    for f in L.functions(mod)
        L.isdeclaration(f) && continue
        acc = Dict{Symbol, Int}()
        for bb in L.blocks(f), inst in L.instructions(bb)
            op = L.opcode(inst)
            t = L.value_type(inst)
            cls = if haskey(fp_arith, op)
                fp64_or_32(fp_arith[op], isdouble(t))
            elseif op == API.LLVMFCmp
                fp64_or_32(:cmp, isdouble(first_operand_type(inst)))
            elseif op in cvt_ops
                fp64_or_32(:cvt, isdouble(t) || isdouble(first_operand_type(inst)))
            elseif op == API.LLVMCall
                callee = L.called_operand(inst)
                name = callee isa L.Function ? L.name(callee) : ""
                if name == "llvm.fma.f64" || name == "llvm.fmuladd.f64"
                    :fp64_fma
                elseif name == "llvm.sqrt.f64"
                    :fp64_sqrt
                elseif startswith(name, "llvm.")
                    dbl = isdouble(t) || any(a -> isdouble(L.value_type(a)), L.arguments(inst))
                    dbl ? :fp64_intrinsic : (isfp(t) ? :fp32 : :other)
                else
                    :call
                end
            elseif op in int_ops
                :int
            elseif op == API.LLVMLoad
                :mem_load
            elseif op == API.LLVMStore
                :mem_store
            elseif op == API.LLVMAtomicRMW || op == API.LLVMAtomicCmpXchg
                :mem_atomic
            elseif op in ctl_ops
                :control
            else
                :other
            end
            acc[cls] = get(acc, cls, 0) + 1
            if op in contractible && isdouble(t) && contract_flag(inst)
                acc[:fp64_contract] = get(acc, :fp64_contract, 0) + 1
            end
        end
        per_function[L.name(f)] = IRCounts(ntuple(i -> get(acc, IR_CLASSES[i], 0), length(IR_CLASSES)))
    end
    return per_function
end

_add_ir(a::IRCounts, b::IRCounts) = IRCounts(ntuple(i -> a[i] + b[i], length(IR_CLASSES)))
_zero_ir() = IRCounts(ntuple(_ -> 0, length(IR_CLASSES)))

"""
    kernel_ir_mix(backend, ck::CompiledKernel; target = nothing) -> NamedTuple

Typed operation count of the OPTIMIZED LLVM IR of a compiled kernel's job (the same job
[`kernel_instruction_mix`](@ref) disassembles; `target` as there), walked with LLVM.jl by
opcode and operand type — the arithmetic before the backend's FMA contraction and div/sqrt
expansion. Returns `counts` (a NamedTuple over [`IR_CLASSES`](@ref), all functions of the
module), `fp64` (its FP64 sum), `total`, `functions` (`Dict` name → counts, for a module that
still has out-of-line callees) and `target`. Whole-module totals: IR loops are not attributed
(the machine-code hot loop is the per-slot figure; the IR is for the fusion/expansion ratio).
"""
function kernel_ir_mix(backend::KA.Backend, ck::CompiledKernel; target = nothing)
    r = _kernel_ir_counts(backend, ck, target === nothing ? nothing : String(target))
    counts = reduce(_add_ir, values(r.functions); init = _zero_ir())
    return (; target = r.isa, total = sum(counts), fp64 = sum(counts[c] for c in IR_FP64_CLASSES), counts,
        functions = r.functions)
end
kernel_ir_mix(::KA.CPU, ck::CompiledKernel; kwargs...) = error("kernel_ir_mix: the CPU backend compiles no GPU kernels")
_kernel_ir_counts(b::KA.Backend, ck::CompiledKernel, target) = error(
    "kernel_ir_mix: no GPU vendor extension loaded for ", typeof(b), " — load CUDA.jl or AMDGPU.jl"
)
