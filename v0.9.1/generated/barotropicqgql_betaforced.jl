using FourierFlows, Plots, Statistics, Printf, Random

using FourierFlows: parsevalsum
using FFTW: irfft
using Random: seed!
using Statistics: mean

import GeophysicalFlows.BarotropicQGQL
import GeophysicalFlows.BarotropicQGQL: energy, enstrophy

dev = CPU()     # Device (CPU/GPU)
nothing # hide

      n = 128            # 2D resolution = n^2
stepper = "FilteredRK4"  # timestepper
     dt = 0.05           # timestep
 nsteps = 8000           # total number of time-steps
 nsubs  = 10             # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide

L = 2π        # domain size
β = 10.0      # planetary PV gradient
μ = 0.01      # bottom drag
nothing # hide

forcing_wavenumber = 14.0    # the central forcing wavenumber for a spectrum that is a ring in wavenumber space
forcing_bandwidth  = 1.5     # the width of the forcing spectrum
ε = 0.001                    # energy input rate by the forcing

grid = TwoDGrid(n, L)

K = @. sqrt(grid.Krsq)                          # a 2D array with the total wavenumber
k = [grid.kr[i] for i=1:grid.nkr, j=1:grid.nl]  # a 2D array with the zonal wavenumber

forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))
@. forcing_spectrum = ifelse(K < 2  * 2π/L, 0, forcing_spectrum)      # no power at low wavenumbers
@. forcing_spectrum = ifelse(K > 20 * 2π/L, 0, forcing_spectrum)      # no power at high wavenumbers
@. forcing_spectrum = ifelse(k < 2π/L, 0, forcing_spectrum)    # make sure forcing does not have power at k=0
ε0 = parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
@. forcing_spectrum *= ε/ε0               # normalize forcing to inject energy at rate ε

seed!(1234) # reset of the random number generator for reproducibility
nothing # hide

function calcF!(Fh, sol, t, clock, vars, params, grid)
  ξ = ArrayType(dev)(exp.(2π * im * rand(eltype(grid), size(sol))) / sqrt(clock.dt))
  @. Fh = ξ * sqrt.(forcing_spectrum)
  @. Fh = ifelse(abs(grid.Krsq) == 0, 0, Fh)

  return nothing
end
nothing # hide

prob = BarotropicQGQL.Problem(dev; nx=n, Lx=L, β=β, μ=μ, dt=dt, stepper=stepper,
                              calcF=calcF!, stochastic=true)
nothing # hide

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = grid.x, grid.y
nothing # hide

calcF!(vars.Fh, sol, 0.0, clock, vars, params, grid)

heatmap(x, y, irfft(vars.Fh, grid.nx)',
     aspectratio = 1,
               c = :balance,
            clim = (-8, 8),
           xlims = (-grid.Lx/2, grid.Lx/2),
           ylims = (-grid.Ly/2, grid.Ly/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "a forcing realization",
      framestyle = :box)

BarotropicQGQL.set_zeta!(prob, zeros(grid.nx, grid.ny))
nothing # hide

E = Diagnostic(energy, prob; nsteps=nsteps)
Z = Diagnostic(enstrophy, prob; nsteps=nsteps)
nothing # hide

function zetaMean(prob)
  sol = prob.sol
  sol[1, :]
end

zMean = Diagnostic(zetaMean, prob; nsteps=nsteps, freq=10)  # the zonal-mean vorticity
nothing # hide

diags = [E, Z, zMean] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide

filepath = "."
plotpath = "./plots_forcedbetaQLturb"
plotname = "snapshots"
filename = joinpath(filepath, "forcedbetaQLturb.jld2")
nothing # hide

if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

get_sol(prob) = sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im * g.l .* g.invKrsq .* sol, g.nx)
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
nothing # hide

function plot_output(prob)
  ζ̄, ζ′= prob.vars.Zeta, prob.vars.zeta
  ζ = @. ζ̄ + ζ′
  ψ̄, ψ′= prob.vars.Psi,  prob.vars.psi
  ψ = @. ψ̄ + ψ′
  ζ̄ₘ = mean(ζ̄, dims=1)'
  ūₘ = mean(prob.vars.U, dims=1)'

  pζ = heatmap(x, y, ζ',
       aspectratio = 1,
            legend = false,
                 c = :balance,
              clim = (-8, 8),
             xlims = (-grid.Lx/2, grid.Lx/2),
             ylims = (-grid.Ly/2, grid.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "vorticity ζ=∂v/∂x-∂u/∂y",
        framestyle = :box)

  pψ = contourf(x, y, ψ',
            levels = -0.32:0.04:0.32,
       aspectratio = 1,
         linewidth = 1,
            legend = false,
              clim = (-0.22, 0.22),
                 c = :viridis,
             xlims = (-grid.Lx/2, grid.Lx/2),
             ylims = (-grid.Ly/2, grid.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "streamfunction ψ",
        framestyle = :box)

  pζm = plot(ζ̄ₘ, y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-3, 3),
            xlabel = "zonal mean ζ",
            ylabel = "y")
  plot!(pζm, 0*y, y, linestyle=:dash, linecolor=:black)

  pum = plot(ūₘ, y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-0.5, 0.5),
            xlabel = "zonal mean u",
            ylabel = "y")
  plot!(pum, 0*y, y, linestyle=:dash, linecolor=:black)

  pE = plot(1,
             label = "energy",
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.1, 4.1),
             ylims = (0, 0.05),
            xlabel = "μt")

  pZ = plot(1,
             label = "enstrophy",
         linecolor = :red,
            legend = :bottomright,
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.1, 4.1),
             ylims = (0, 5),
            xlabel = "μt")

  l = @layout Plots.grid(2, 3)
  p = plot(pζ, pζm, pE, pψ, pum, pZ, layout=l, size = (1000, 600), dpi=150)

  return p
end

p = plot_output(prob)

startwalltime = time()

anim = @animate for j = 0:round(Int, nsteps / nsubs)

  if j % (1000 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.v) / grid.dy, maximum(vars.u .+ vars.U) / grid.dx])

    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
      clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i],
      (time()-startwalltime)/60)

    println(log)
  end

  p[1][1][:z] = @. vars.zeta + vars.Zeta
  p[1][:title] = "vorticity, μt="*@sprintf("%.2f", μ * clock.t)
  p[4][1][:z] = @. vars.psi + vars.Psi
  p[2][1][:x] = mean(vars.Zeta, dims=1)'
  p[5][1][:x] = mean(vars.U, dims=1)'
  push!(p[3][1], μ * E.t[E.i], E.data[E.i])
  push!(p[6][1], μ * Z.t[Z.i], Z.data[Z.i])

  stepforward!(prob, diags, nsubs)
  BarotropicQGQL.updatevars!(prob)
end

mp4(anim, "barotropicqgql_betaforced.mp4", fps=18)

savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), clock.step)
savefig(savename)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

