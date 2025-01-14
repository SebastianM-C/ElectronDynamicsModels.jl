@connector function ElectromagneticField(; name, t = τ)
    @variables E(t)[1:3] [input = true] B(t)[1:3] [input = true]
    @variables Fμν(t)[1:4, 1:4] [input = true]

    @parameters c = austrip(c_0), [tunable = false]

    eqs = [
        # contravariant formulation
        Fμν ~ [
            0 -E[1]/c -E[2]/c -E[3]/c
            E[1]/c 0 -B[3] B[2]
            E[2]/c B[3] 0 -B[1]
            E[3]/c -B[2] B[1] 0
        ],
    ]

    ODESystem(eqs, t; name)
end

function GaussLaser(; name)
    @parameters begin
        λ = austrip(800u"nm")
        a₀ = 10.0
        c = austrip(c_0), [tunable = false]
        m = austrip(m_e)
        q = austrip(e)
        ω
        k
        E₀
        w₀ = 75λ
        z_R
        T0 = 100 # 5 * austrip(c_0 / λ)
        τ0 = 10 / ω
    end

    ξx = 1.0 + 0im
    ξy = 0
    ϕ₀ = 0
    t₀ = 5T0
    z₀ = 0

    @variables E(τ)[1:3] [output = true] B(τ)[1:3] [output = true]

    @variables t(τ) x(τ)[1:4] [input = true] wz(τ) z(τ) r(τ) y(t) z(τ)

    eqs = [
        z ~ x[4]
        wz ~ w₀ * √(1 + (z / z_R)^2)
        r ~ hypot(x[2], x[3])
        E[1] ~ real(
            E₀ * w₀ / wz *
            exp(
                -(r / wz)^2 + im * (-(r^2 * z) / (z_R * wz^2) + atan(z, z_R) - k * z + ϕ₀),
            ) *
            exp(im * ω * t) *
            exp(-(((t - t₀) - (z - z₀) / c) / τ0)^2),
        )
        E[2] ~ 0
        E[3] ~ real(
            2im / (k * wz^2) *
            (1 + im * (z / z_R)) *
            (x[2] * E[1] + x[3] * E[2]) *
            exp(im * ω * t) *
            exp(-(((t - t₀) - (z - z₀) / c) / τ0)^2),
        )
        B[1] ~ 0
        B[2] ~ 0
        B[3] ~ real(
            2im / (k * c * wz^2) *
            (1 + im * (z / z_R)) *
            (x[3] * E[1] - x[2] * E[2]) *
            exp(im * ω * t) *
            exp(-(((t - t₀) - (z - z₀) / c) / τ0)^2),
        )
    ]

    ODESystem(
        eqs,
        τ;
        name,
        parameter_dependencies = [
            ω ~ 2π * c / λ,
            E₀ ~ a₀ * m * c * ω / abs(q),
            z_R ~ w₀^2 * k / 2,
            k ~ 2π / λ,
        ],
    )
end

function PlaneWave(; name)
    @parameters begin
        λ = austrip(800u"nm")
        a₀ = 10.0
        c = austrip(c_0), [tunable = false]
        m = austrip(m_e)
        q = austrip(e)
        ω
        k
        E₀
    end

    @variables E(τ)[1:3] [output = true] B(τ)[1:3] [output = true]
    @variables t(τ) x(τ)[1:4] [input = true]

    eqs = [
        E[1] ~ E₀ * cos(k * x[4] - ω * t)
        E[2] ~ 0
        E[3] ~ 0
        B[1] ~ 0
        B[2] ~ 1 / c * E₀ * cos(k * x[4] - ω * t)
        B[3] ~ 0
    ]

    ODESystem(
        eqs,
        τ;
        name,
        parameter_dependencies = [ω ~ 2π * c / λ, k ~ 2π / λ, E₀ ~ a₀ * m * c * ω / abs(q)],
    )
end

function PlaneWave2(; name)
    @parameters begin
        λ = austrip(800u"nm")
        a₀ = 10.0
        c = austrip(c_0), [tunable = false]
        m = austrip(m_e)
        q = austrip(e)
        ω
        k
        E₀
    end

    @independent_variables t

    @variables E(t)[1:3] [output = true] B(t)[1:3] [output = true]
    @variables τ(t) x(t)[1:4] [input = true]

    eqs = [
        E[1] ~ E₀ * cos(k * x[4] - ω * t)
        E[2] ~ 0
        E[3] ~ 0
        B[1] ~ 0
        B[2] ~ 1 / c * E₀ * cos(k * x[4] - ω * t)
        B[3] ~ 0
    ]

    ODESystem(
        eqs,
        t;
        name,
        parameter_dependencies = [ω ~ 2π * c / λ, k ~ 2π / λ, E₀ ~ a₀ * m * c * ω / abs(q)],
    )
end
