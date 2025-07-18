function Spacetime(; name, c)
    @parameters c=c gμν[1:4, 1:4] = diagm([1, -1, -1, -1])
    System(Equation[], τ, [], GlobalScope.([c, gμν]); name)
end

function ElectromagneticSystem(; name)
    @variables begin
        Fμν(τ)[1:4, 1:4], [description = "Faraday tensor"]
        T(τ)[1:4, 1:4], [description="Stress-energy tensor T^μν"]
        J(τ)[1:4], [description="4-current density"]
    end
    System(Equation[], τ, [Fμν, T, J], []; name)
end

