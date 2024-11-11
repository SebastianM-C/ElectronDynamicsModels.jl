using ElectronDynamicsModels
using Documenter

DocMeta.setdocmeta!(ElectronDynamicsModels, :DocTestSetup, :(using ElectronDynamicsModels); recursive=true)

makedocs(;
    modules=[ElectronDynamicsModels],
    authors="Sebastian Micluța-Câmpeanu <sebastian.mc95@proton.me> and contributors",
    sitename="ElectronDynamicsModels.jl",
    format=Documenter.HTML(;
        canonical="https://SebastianM-C.github.io/ElectronDynamicsModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/SebastianM-C/ElectronDynamicsModels.jl",
    devbranch="main",
)
