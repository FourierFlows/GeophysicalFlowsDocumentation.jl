using FourierFlows, Printf, Plots

using FourierFlows: parsevalsum
using Random: seed!
using FFTW: irfft

import GeophysicalFlows.TwoDNavierStokes
import GeophysicalFlows.TwoDNavierStokes: energy, energy_dissipation, energy_work, energy_drag
import GeophysicalFlows.TwoDNavierStokes: enstrophy, enstrophy_dissipation, enstrophy_work, enstrophy_drag

dev = CPU()    # Device (CPU/GPU)
nothing # hide

 n, L  = 256, 2π              # grid resolution and domain length
 ν, nν = 2e-7, 2              # hyperviscosity coefficient and hyperviscosity order
 μ, nμ = 1e-1, 0              # linear drag coefficient
dt, tf = 0.005, 0.2 / μ       # timestep and final time
    nt = round(Int, tf / dt)  # total timesteps
    ns = 4                    # how many intermediate times we want to plot
nothing # hide

forcing_wavenumber = 14.0    # the central forcing wavenumber for a spectrum that is a ring in wavenumber space
forcing_bandwidth  = 1.5     # the width of the forcing spectrum
ε = 0.1                      # energy input rate by the forcing

grid = TwoDGrid(dev, n, L)

K = @. sqrt(grid.Krsq)
forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))
forcing_spectrum[K .< ( 2 * 2π/L)] .= 0 # no power at low wavenumbers
forcing_spectrum[K .> (20 * 2π/L)] .= 0 # no power at high wavenumbers
ε0 = parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
@. forcing_spectrum *= ε/ε0             # normalize forcing to inject energy at rate ε

seed!(1234)
nothing # hide

function calcF!(Fh, sol, t, clock, vars, params, grid)
  ξ = ArrayType(dev)(exp.(2π * im * rand(eltype(grid), size(sol))) / sqrt(clock.dt))
  ξ[1, 1] = 0
  @. Fh = ξ * sqrt(forcing_spectrum)
  nothing
end
nothing # hide

prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt, stepper="ETDRK4",
                                calcF=calcF!, stochastic=true)
nothing # hide

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

x, y = grid.x, grid.y
nothing # hide

calcF!(vars.Fh, sol, 0.0, clock, vars, params, grid)

heatmap(x, y, irfft(vars.Fh, grid.nx),
     aspectratio = 1,
               c = :balance,
            clim = (-200, 200),
           xlims = (-L/2, L/2),
           ylims = (-L/2, L/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "a forcing realization",
      framestyle = :box)

TwoDNavierStokes.set_zeta!(prob, zeros(grid.nx, grid.ny))

E  = Diagnostic(energy,                prob, nsteps=nt) # energy
Rᵋ = Diagnostic(energy_drag,           prob, nsteps=nt) # energy dissipation by drag
Dᵋ = Diagnostic(energy_dissipation,    prob, nsteps=nt) # energy dissipation by hyperviscosity
Wᵋ = Diagnostic(energy_work,           prob, nsteps=nt) # energy work input by forcing
Z  = Diagnostic(enstrophy,             prob, nsteps=nt) # enstrophy
Rᶻ = Diagnostic(enstrophy_drag,        prob, nsteps=nt) # enstrophy dissipation by drag
Dᶻ = Diagnostic(enstrophy_dissipation, prob, nsteps=nt) # enstrophy dissipation by hyperviscosity
Wᶻ = Diagnostic(enstrophy_work,        prob, nsteps=nt) # enstrophy work input by forcing
diags = [E, Dᵋ, Wᵋ, Rᵋ, Z, Dᶻ, Wᶻ, Rᶻ] # a list of Diagnostics passed to `stepforward!` will  be updated every timestep.
nothing # hide

function computetendencies_and_makeplot(prob, diags)
  sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

  TwoDNavierStokes.updatevars!(prob)

  E, Dᵋ, Wᵋ, Rᵋ, Z, Dᶻ, Wᶻ, Rᶻ = diags

  clocktime = round(μ * clock.t, digits=2)

  dEdt_numerical = (E[2:E.i] - E[1:E.i-1]) / clock.dt # numerical first-order approximation of energy tendency
  dZdt_numerical = (Z[2:Z.i] - Z[1:Z.i-1]) / clock.dt # numerical first-order approximation of enstrophy tendency

  dEdt_computed = Wᵋ[2:E.i] - Dᵋ[1:E.i-1] - Rᵋ[1:E.i-1]
  dZdt_computed = Wᶻ[2:Z.i] - Dᶻ[1:Z.i-1] - Rᶻ[1:Z.i-1]

  residual_E = dEdt_computed - dEdt_numerical
  residual_Z = dZdt_computed - dZdt_numerical

  εᶻ = parsevalsum(forcing_spectrum / 2, grid) / (grid.Lx * grid.Ly)

  pzeta = heatmap(x, y, vars.zeta,
            aspectratio = 1,
            legend = false,
                 c = :viridis,
              clim = (-25, 25),
             xlims = (-L/2, L/2),
             ylims = (-L/2, L/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "μt",
            ylabel = "y",
             title = "∇²ψ(x, y, μt="*@sprintf("%.2f", μ*clock.t)*")",
        framestyle = :box)

  pζ = plot(pzeta, size = (400, 400))

  t = E.t[2:E.i]

  p1E = plot(μ * t, [Wᵋ[2:E.i] ε.+0*t -Dᵋ[1:E.i-1] -Rᵋ[1:E.i-1]],
             label = ["energy work, Wᵋ" "ensemble mean energy work, <Wᵋ>" "dissipation, Dᵋ" "drag, Rᵋ = - 2μE"],
         linestyle = [:solid :dash :solid :solid],
         linewidth = 2,
             alpha = 0.8,
            xlabel = "μt",
            ylabel = "energy sources and sinks")

  p2E = plot(μ * t, [dEdt_computed, dEdt_numerical],
           label = ["computed Wᵋ-Dᵋ" "numerical dE/dt"],
       linestyle = [:solid :dashdotdot],
       linewidth = 2,
           alpha = 0.8,
          xlabel = "μt",
          ylabel = "dE/dt")

  p3E = plot(μ * t, residual_E,
           label = "residual dE/dt = computed - numerical",
       linewidth = 2,
           alpha = 0.7,
          xlabel = "μt")

  t = Z.t[2:E.i]

  p1Z = plot(μ * t, [Wᶻ[2:Z.i] εᶻ.+0*t -Dᶻ[1:Z.i-1] -Rᶻ[1:Z.i-1]],
           label = ["enstrophy work, Wᶻ" "mean enstrophy work, <Wᶻ>" "enstrophy dissipation, Dᶻ" "enstrophy drag, Rᶻ = - 2μZ"],
       linestyle = [:solid :dash :solid :solid],
       linewidth = 2,
           alpha = 0.8,
          xlabel = "μt",
          ylabel = "enstrophy sources and sinks")


  p2Z = plot(μ * t, [dZdt_computed, dZdt_numerical],
         label = ["computed Wᶻ-Dᶻ" "numerical dZ/dt"],
     linestyle = [:solid :dashdotdot],
     linewidth = 2,
         alpha = 0.8,
        xlabel = "μt",
        ylabel = "dZ/dt")

  p3Z = plot(μ * t, residual_Z,
         label = "residual dZ/dt = computed - numerical",
     linewidth = 2,
         alpha = 0.7,
        xlabel = "μt")

  layout = @layout Plots.grid(3, 2)

  pbudgets = plot(p1E, p1Z, p2E, p2Z, p3E, p3Z, layout=layout, size = (900, 1200))

  return pζ, pbudgets
end
nothing # hide

startwalltime = time()
for i = 1:ns
  stepforward!(prob, diags, round(Int, nt/ns))

  TwoDNavierStokes.updatevars!(prob)

  cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

  log = @sprintf("step: %04d, t: %.1f, cfl: %.3f, walltime: %.2f min", clock.step, clock.t,
        cfl, (time()-startwalltime)/60)

  println(log)
end

pζ, pbudgets = computetendencies_and_makeplot(prob, diags)

pζ

pbudgets

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

