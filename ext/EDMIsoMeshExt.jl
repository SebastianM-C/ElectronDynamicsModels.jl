# Marching-cubes isosurface → GeometryBasics.Mesh (see src/rpr_api.jl for the
# docstring). Separate from the RPRMakie extension because it has no RPR
# dependency — the GLMakie interactive path uses the same meshes.
module EDMIsoMeshExt

import ElectronDynamicsModels as EDM
using MarchingCubes: MC, march
import MarchingCubes
import GeometryBasics

function EDM.iso_mesh(vol, Xs, Ys, Zs, level)
    m = MC(vol; x = collect(Float32, Xs), y = collect(Float32, Ys),
        z = collect(Float32, Zs))
    march(m, level)
    isempty(m.triangles) && return nothing
    return MarchingCubes.makemesh(GeometryBasics, m)
end

end
