using Pkg
Pkg.activate(@__DIR__)

using Typst_jll

# Change to poster directory
cd(@__DIR__)

# Compile the poster
println("Compiling poster...")
run(`$(Typst_jll.typst()) compile poster.typ poster.pdf`)
println("✓ Poster compiled to poster.pdf")
