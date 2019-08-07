#
# # DATE: 2011-09-15 CONTRIBUTOR: Unknown CITATION: Justo, Bazant, Kaxiras, Bulatov and Yip, Phys Rev B, 58, 2539 (1998)
#
# # EDIP parameters for various elements and mixtures
# # multiple entries can be added to this file, LAMMPS reads the ones it needs
# # these entries are in LAMMPS "metal" units
#
# # format of a single entry (one or more lines)
# #
# #   element 1, element 2, element 3,
# #     A   B   cutoffA   cutoffC   alpha   beta   eta
# #     gamma   lambda    mu   rho   sigma   Q0
# #     u1   u2   u3   u4
# #
# # units for each parameters:
# #     A , lambda are in eV
# #     B, cutoffA, cutoffC, gamma, sigma are in Angstrom
# #     alpha, beta, eta, mu, rho, Q0, u1-u4 are pure numbers
#
# # Here are the original parameters in metal units, for Silicon from:
# # J. F. Justo, M. Z. Bazant, E. Kaxiras, V. V. Bulatov, S. Yip
# #       Phys. Rev. B 58, 2539 (1998)
# #
#
# Si Si Si 7.9821730 1.5075463 3.1213820 2.5609104 3.1083847 0.0070975 0.2523244
#          1.1247945 1.4533108 0.6966326 1.2085196 0.5774108 312.1341346
#          -0.165799 32.557 0.286198 0.66


module EDIP

using JuLIP
JuLIP.Potentials.WrappedAnalyticFunction
import JuLIP
import JuLIP.Potentials: SitePotential, WrappedAnalyticFunction

struct EdIP <: SitePotential
   neigf::WrappedAnalyticFunction
end

function EdIP(;
                  A        = 7.9821730
                  B        = 1.5075463
                  cutoffA  = 3.1213820   # a = cutoffA
                  cutoffC  = 2.5609104   # c = cutoffC
                  alpha    = 3.1083847
                  beta     = 0.0070975
                  eta      = 0.2523244
                  gamma    = 1.1247945
                  lambda   = 1.4533108
                  mu       = 0.6966326
                  rho      = 1.2085196
                  sigma    = 0.5774108
                  Q0       = 312.1341346
                  u1       = -0.165799
                  u2       = 32.557
                  u3       = 0.286198
                  u4       = 0.66 )
   neigf = let a = cutoffA, c = cutoffC, α = alpha
      @analytic x -> exp(α / (1 - 1 / x^3))
   end
   moll2 = let σ = sigma, a = cutoffA, A = A
      @analytic r -> A *  exp( σ / (r - a) )
   end
   moll3 = let γ = gamma, a = cutoffA
      @analytic r -> exp( γ / (r - a) )
   end
   pf = let β = beta
      @analytic Z -> exp( - β * Z^2 )
   end
   v2in = let B = B, ρ = rho
      @analytic r -> (B / r)^ρ
   end
end


function neigf(V::EdIP, r)
   x = (r - V.cutoffC) / (V.cutoffA - V.cutoffC)
   if x <= 1e-7
      return 1.0
   elseif x >= 1 - 2.5e-2
      return 0.0
   else
      return V.neigf(x)
   end
end

function neigf_d(V::EdIP, r)
   x = (r - V.cutoffC) / (V.cutoffA - V.cutoffC)
   dxdr = 1 / (V.cutoffA - V.cutoffC)
   if (x <= 1e-7) || (x >= 1 - 2.5e-2)
      return 0.0
   else
      return (@D V.neigf(x)) * dxdr
   end
end

V2(V::EdIP, R, Z) = (V.v2in(norm(R)) - V.pf(Z)) * V.moll2(norm(R))

function  V2_d(V::EdIP, R, Z)   # -> ∂V2/∂R, ∂V2/∂Z
   r = norm(R)
   v2in = V.v2in(r)
   p = V.pf(Z)
   moll2 = V.moll2(r)
   DV2Dr = (@D V.v2in(r)) * moll2 + (v2in - p) * (@D V.moll2(r))
   return (DV2Dr / r) *  R, (- moll2) * (@D V.pf(Z))
end

function V3(V::EdIP, R1, R2, Z)
   r1 = norm(R1)
   r2 = norm(R2)
   g1 = V.moll3(r1)
   g2 = V.moll3(r2)
   l12 = dot(R1, R2) / (r1*r2)
   h = hf(V, l12, Z)
   return g1 * g2 * h
end



function evaluate!(tmp, V::EdIP, R::AbstractVector{<: JVec})

   # compute the neighbour counter + the r's along the way
   Zi = 0.0
   for j = 1:length(R)
      Zi += neigf(V, norm(R[j]))
   end

   # compute the V2's
   Es = 0.0
   for j = 1:length(R)
      Es += V2(V, R[j], Zi)
   end

   # compute the V3's
   for j1 = 2:length(R), j2 = 1:j1-1
      l12 = dot(R[j1], R[j2]) / (tmp.r[j1]*tmp.r[j2])
      Es += V3(V, R[j1], R[j2], Zi)
   end
end
