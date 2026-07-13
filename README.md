# ElectronDynamicsModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SebastianM-C.github.io/ElectronDynamicsModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SebastianM-C.github.io/ElectronDynamicsModels.jl/dev/)
[![Build Status](https://github.com/SebastianM-C/ElectronDynamicsModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/SebastianM-C/ElectronDynamicsModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/SebastianM-C/ElectronDynamicsModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SebastianM-C/ElectronDynamicsModels.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This package aims to provide several models for electron dynamics such that they can be used within the ModelingToolkit
modeling framework in a composable way, such that one can better understand the impact of various physical phenomena
such as radiation reaction or photon emmision.

![Ray-traced Thomson-scattering animation: a vortex laser pulse rings a disk of electrons, the radiated
wavefronts propagate to a detector, and the recorded fluence develops into the OAM annulus](animation/thomson_rpr.gif)

*Thomson scattering, exactly as computed: an OAM laser pulse (red/blue wavefronts) crosses a disk of electrons;
their radiated field (gold/violet, signed Liénard–Wiechert far field) propagates to the detector, which records
the fluence — the dark spot at the center of the developed ring is the optical-vortex singularity.
Ray-traced with [RPRMakie](https://docs.makie.org/stable/documentation/backends/rprmakie/); see `animation/`.*
