function ReferenceFrame(; name, c, ε₀, μ₀)
    @parameters gμν[1:4, 1:4] = diagm([1, -1, -1, -1])
    constants = @parameters c=c ε₀=ε₀ μ₀=μ₀
    System(Equation[], GlobalScope(τ), [], GlobalScope.([constants..., gμν]); name)
end

function ElectromagneticSystem(; name)
    @variables begin
        Fμν(τ)[1:4, 1:4], [description = "Faraday tensor"]
        T(τ)[1:4, 1:4], [description="Stress-energy tensor T^μν"]
        J(τ)[1:4], [description="4-current density"]
    end
    System(Equation[], τ, [Fμν, T, J], []; name)
end
