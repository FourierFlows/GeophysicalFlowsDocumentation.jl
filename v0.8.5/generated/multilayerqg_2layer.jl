using FourierFlows, Plots, Printf

using FFTW: rfft, irfft
import GeophysicalFlows.MultilayerQG
import GeophysicalFlows.MultilayerQG: energies

dev = CPU()     # Device (CPU/GPU)
nothing # hide

nx = 128        # 2D resolution = nx^2
ny = nx

stepper = "FilteredRK4"  # timestepper
     dt = 6e-3           # timestep
 nsteps = 7000           # total number of time-steps
 nsubs  = 25             # number of time-steps for plotting (nsteps must be multiple of nsubs)
nothing # hide

Lx = 2π         # domain size
 μ = 5e-2       # bottom drag
 β = 5          # the y-gradient of planetary PV

nlayers = 2      # number of layers
f₀, g = 1, 1     # Coriolis parameter and gravitational constant
 H = [0.2, 0.8]  # the rest depths of each layer
 ρ = [4.0, 5.0]  # the density of each layer

 U = zeros(nlayers) # the imposed mean zonal flow in each layer
 U[1] = 1.0
 U[2] = 0.0
nothing # hide

prob = MultilayerQG.Problem(nlayers, dev; nx=nx, Lx=Lx, f₀=f₀, g=g, H=H, ρ=ρ, U=U, dt=dt, stepper=stepper, μ=μ, β=β)
nothing # hide

sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
x, y = gr.x, gr.y
nothing # hide

q_i  = 4e-3randn((nx, ny, nlayers))
qh_i = prob.timestepper.filter .* rfft(q_i, (1, 2)) # only apply rfft in dims=1, 2
q_i  = irfft(qh_i, gr.nx, (1, 2)) # only apply irfft in dims=1, 2

MultilayerQG.set_q!(prob, q_i)
nothing # hide

E = Diagnostic(energies, prob; nsteps=nsteps)
diags = [E] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide

filepath = "."
plotpath = "./plots_2layer"
plotname = "snapshots"
filename = joinpath(filepath, "2layer.jld2")
nothing # hide

if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

get_sol(prob) = sol # extracts the Fourier-transformed solution
function get_u(prob)
  @. v.qh = sol
  streamfunctionfrompv!(v.ψh, v.qh, p, g)
  @. v.uh = -im * g.l * v.ψh
  invtransform!(v.u, v.uh, p)
  return v.u
end
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
nothing #hide

symlims(data) = maximum(abs.(extrema(data))) |> q -> (-q, q)

function plot_output(prob)

  l = @layout grid(2, 3)
  p = plot(layout=l, size = (1000, 600), dpi=150)

  for m in 1:nlayers
    heatmap!(p[(m-1)*3+1], x, y, vs.q[:, :, m],
         aspectratio = 1,
              legend = false,
                   c = :balance,
               xlims = (-gr.Lx/2, gr.Lx/2),
               ylims = (-gr.Ly/2, gr.Ly/2),
               clims = symlims,
              xticks = -3:3,
              yticks = -3:3,
              xlabel = "x",
              ylabel = "y",
               title = "q_"*string(m),
          framestyle = :box)

    contourf!(p[(m-1)*3+2], x, y, vs.ψ[:, :, m],
              levels = 8,
         aspectratio = 1,
              legend = false,
                   c = :viridis,
               xlims = (-gr.Lx/2, gr.Lx/2),
               ylims = (-gr.Ly/2, gr.Ly/2),
               clims = symlims,
              xticks = -3:3,
              yticks = -3:3,
              xlabel = "x",
              ylabel = "y",
               title = "ψ_"*string(m),
          framestyle = :box)
  end

  plot!(p[3], 2,
             label = ["KE1" "KE2"],
            legend = :bottomright,
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.1, 2.35),
             ylims = (5e-10, 1e0),
            yscale = :log10,
            yticks = 10.0.^(-9:2:0),
            xlabel = "μt")

  plot!(p[6], 1,
             label = "PE",
            legend = :bottomright,
         linecolor = :red,
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.1, 2.35),
             ylims = (1e-10, 1e0),
            yscale = :log10,
            yticks = 10.0.^(-10:2:0),
            xlabel = "μt")

end
nothing # hide

p = plot_output(prob)

startwalltime = time()

anim = @animate for j=0:Int(nsteps/nsubs)

  cfl = cl.dt*maximum([maximum(vs.u)/gr.dx, maximum(vs.v)/gr.dy])

  log = @sprintf("step: %04d, t: %.1f, cfl: %.2f, KE1: %.4f, KE2: %.4f, PE: %.4f, walltime: %.2f min", cl.step, cl.t, cfl, E.data[E.i][1][1], E.data[E.i][1][2], E.data[E.i][2][1], (time()-startwalltime)/60)

  if j%(1000/nsubs)==0; println(log) end

  for m in 1:nlayers
    p[(m-1)*3+1][1][:z] = @. vs.q[:, :, m]
    p[(m-1)*3+2][1][:z] = @. vs.ψ[:, :, m]
  end

  push!(p[3][1], μ*E.t[E.i], E.data[E.i][1][1])
  push!(p[3][2], μ*E.t[E.i], E.data[E.i][1][2])
  push!(p[6][1], μ*E.t[E.i], E.data[E.i][2][1])

  stepforward!(prob, diags, nsubs)
  MultilayerQG.updatevars!(prob)
end

mp4(anim, "multilayerqg_2layer.mp4", fps=18)

savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
savefig(savename)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

