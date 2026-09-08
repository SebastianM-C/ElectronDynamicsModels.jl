# GPM (GPU Performance Monitoring) child sampler over NVML — spawned by `with_gpm_sampler`
# (src/gpm.jl) through the CUDA extension's `gpu_gpm_child_cmd`. Runs OUT OF PROCESS for the
# same reason as bin/gputrace*.sh (an in-process Julia sampler is suspended by the parent's
# GC/timer coupling), and as a JULIA child because `nvidia-smi` has no GPM query: this script
# loads CUDA.jl for its NVML bindings ONLY — it never creates a CUDA context or touches a device.
#
# Usage: julia --project=<env with CUDA> gpmtrace.jl <dt_s> <parent_pid> <stopfile> <uuid=ordinal>...
#   uuid = the GPU-… NVML uuid (stable under CUDA_VISIBLE_DEVICES); ordinal = the parent's 1-based id
#
# Output on STDOUT (the parent redirects it into the gpmtrace TSV): first a header comment naming
# the columns, then one row per device per tick:
#   epoch_s <TAB> device <TAB> sm_util <TAB> sm_occupancy <TAB> fp64_util <TAB> dram_bw_util <TAB>
#   fp32_util <TAB> fp16_util <TAB> tensor_util <TAB> int_util <TAB> pcie_tx_MiBps <TAB> pcie_rx_MiBps <TAB>
#   nvlink_rx_MiBps <TAB> nvlink_tx_MiBps
# Utilizations/occupancy are fractions of the interval-average in [0, 1] (NVML reports percent);
# traffic columns are MiB/s as NVML defines them; `nan` = the device reports no such metric.
#
# NVML GPM model: a metric is the difference of two samples over their interval. Each tick takes
# ONE new sample per device and evaluates the metrics against the previous tick's sample, so a
# row covers exactly the preceding `dt` seconds (the first row appears after the second sample).
# Exits when <stopfile> appears or the parent dies — cooperative stop, no signals, no orphans.

length(ARGS) >= 4 || (println(stderr, "usage: gpmtrace.jl <dt_s> <parent_pid> <stopfile> <uuid=ordinal>..."); exit(64))
const DT = parse(Float64, ARGS[1])
const PPID = parse(Int, ARGS[2])
const STOPFILE = ARGS[3]
const SPECS = ARGS[4:end]

# Declare the columns before the (seconds-long) package load so a stalled child still leaves a
# header the parent can recognise.
const METRICS = (   # column name => NVML GPM metric id (nvmlGpmMetricId_t), unit scale
    (:sm_util, 2, 0.01), (:sm_occupancy, 3, 0.01), (:fp64_util, 11, 0.01), (:dram_bw_util, 10, 0.01),
    (:fp32_util, 12, 0.01), (:fp16_util, 13, 0.01), (:tensor_util, 5, 0.01), (:int_util, 4, 0.01),
    (:pcie_tx_MiBps, 20, 1.0), (:pcie_rx_MiBps, 21, 1.0), (:nvlink_rx_MiBps, 60, 1.0), (:nvlink_tx_MiBps, 61, 1.0),
)
println("# epoch_s\tdevice\t", join((String(m[1]) for m in METRICS), '\t'))
flush(stdout)

using CUDA: CUDA
const NVML = CUDA.NVML

parent_alive() = ccall(:kill, Cint, (Cint, Cint), PPID, 0) == 0 || Libc.errno() == Libc.EPERM
stop_requested() = isfile(STOPFILE) || !parent_alive()

# nvmlGpmMetricsGet_t is a ~19 kB struct with a fixed-size metric array that CUDA.jl's bindings
# expose as an opaque NTuple; drive it through a byte buffer and field offsets instead.
const GetT = NVML.nvmlGpmMetricsGet_t
const MetricT = NVML.nvmlGpmMetric_t
const OFF_VERSION = fieldoffset(GetT, 1)
const OFF_NUM = fieldoffset(GetT, 2)
const OFF_S1 = fieldoffset(GetT, 3)
const OFF_S2 = fieldoffset(GetT, 4)
const OFF_METRICS = fieldoffset(GetT, 5)

struct Dev
    ordinal::Int
    handle::NVML.nvmlDevice_t
    samples::Vector{NVML.nvmlGpmSample_t}   # two buffers, swapped every tick
end

function open_device(spec::AbstractString)
    uuid, ord = split(spec, '='; limit = 2)
    href = Ref{NVML.nvmlDevice_t}()
    NVML.nvmlDeviceGetHandleByUUID(String(uuid), href)
    sup = Ref(NVML.nvmlGpmSupport_t(NVML.NVML_GPM_SUPPORT_VERSION, 0))
    NVML.nvmlGpmQueryDeviceSupport(href[], sup)
    if sup[].isSupportedDevice == 0
        println(stderr, "gpmtrace: device $ord ($uuid) has no GPM support — skipped")
        return nothing
    end
    samples = map(1:2) do _
        s = Ref{NVML.nvmlGpmSample_t}()
        NVML.nvmlGpmSampleAlloc(s)
        s[]
    end
    return Dev(parse(Int, ord), href[], samples)
end

function metrics_row!(buf::Vector{UInt8}, d::Dev, older::NVML.nvmlGpmSample_t, newer::NVML.nvmlGpmSample_t)
    fill!(buf, 0)
    vals = fill(NaN, length(METRICS))
    GC.@preserve buf begin
        p = pointer(buf)
        unsafe_store!(Ptr{Cuint}(p + OFF_VERSION), Cuint(NVML.NVML_GPM_METRICS_GET_VERSION))
        unsafe_store!(Ptr{Cuint}(p + OFF_NUM), Cuint(length(METRICS)))
        unsafe_store!(Ptr{NVML.nvmlGpmSample_t}(p + OFF_S1), older)
        unsafe_store!(Ptr{NVML.nvmlGpmSample_t}(p + OFF_S2), newer)
        for (i, m) in enumerate(METRICS)
            mp = Ptr{MetricT}(p + OFF_METRICS + (i - 1) * sizeof(MetricT))
            unsafe_store!(mp.metricId, Cuint(m[2]))
        end
        try
            NVML.nvmlGpmMetricsGet(Ptr{GetT}(p))
            for (i, m) in enumerate(METRICS)
                mp = Ptr{MetricT}(p + OFF_METRICS + (i - 1) * sizeof(MetricT))
                unsafe_load(mp.nvmlReturn) == NVML.NVML_SUCCESS || continue
                v = unsafe_load(mp.value)
                isfinite(v) || continue
                vals[i] = v * m[3]
            end
        catch err
            println(stderr, "gpmtrace: nvmlGpmMetricsGet failed on device $(d.ordinal): ", sprint(showerror, err))
        end
    end
    return vals
end

function main()
    devs = Dev[]
    for spec in SPECS
        try
            d = open_device(spec)
            d === nothing || push!(devs, d)
        catch err
            println(stderr, "gpmtrace: cannot open $spec: ", sprint(showerror, err))
        end
    end
    isempty(devs) && (println(stderr, "gpmtrace: no GPM-capable device — exiting"); return)
    buf = zeros(UInt8, sizeof(GetT))
    for d in devs
        NVML.nvmlGpmSampleGet(d.handle, d.samples[1])
    end
    older = 1
    while !stop_requested()
        sleep(DT)
        newer = 3 - older
        now = time()
        for d in devs
            try
                NVML.nvmlGpmSampleGet(d.handle, d.samples[newer])
            catch err
                println(stderr, "gpmtrace: nvmlGpmSampleGet failed on device $(d.ordinal): ", sprint(showerror, err))
                continue
            end
            vals = metrics_row!(buf, d, d.samples[older], d.samples[newer])
            print(round(now; digits = 2), '\t', d.ordinal)
            for (m, v) in zip(METRICS, vals)
                if isnan(v)
                    print("\tnan")
                elseif m[3] == 1.0
                    print('\t', round(v; digits = 1))
                else
                    print('\t', round(v; digits = 4))
                end
            end
            println()
        end
        flush(stdout)
        older = newer
    end
    for d in devs, s in d.samples
        try
            NVML.nvmlGpmSampleFree(s)
        catch
        end
    end
end

main()
