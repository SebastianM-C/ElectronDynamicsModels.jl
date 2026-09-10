using ElectronDynamicsModels
using Documenter
using Documenter.Remotes: GitHub

DocMeta.setdocmeta!(ElectronDynamicsModels, :DocTestSetup, :(using ElectronDynamicsModels); recursive=true)

makedocs(;
    modules=[ElectronDynamicsModels],
    # index.md autodocs GPUDiagnostics too; it is a git-URL dependency (not a checkout), so
    # Documenter cannot infer its remote for source links.
    remotes=Dict(pkgdir(ElectronDynamicsModels.GPUDiagnostics) => (GitHub("SebastianM-C", "GPUDiagnostics.jl"), "main")),
    authors="Sebastian Micluța-Câmpeanu <sebastian.mc95@proton.me> and contributors",
    sitename="ElectronDynamicsModels.jl",
    format=Documenter.HTML(;
        canonical="https://SebastianM-C.github.io/ElectronDynamicsModels.jl",
        edit_link="main",
        assets=String[],
        # index.md autodocs the whole package (+ GPUDiagnostics.jl); it passed 200 KiB with the
        # rocprofv3 docstrings — Documenter's default hard limit.
        size_threshold=400 * 2^10,
        size_threshold_warn=250 * 2^10,
    ),
    pages=[
        "Home" => "index.md",
        "Experimental" => "experimental.md",
        "The device spline" => "gpu_spline.md",
        "Electron batching" => "electron_batches.md",
    ],
)

deploydocs(;
    repo="github.com/SebastianM-C/ElectronDynamicsModels.jl",
    devbranch="main",
)
