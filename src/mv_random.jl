# The code in this file generates multi-variate random distributions with 
# known mean and pair-wise correlation matrix, resulting in binary spiketrains
# with the associated statistics.

# The code immediately below is a reduced implementation of the Bivariate Gaussian code
# available at https://github.com/mschauer/GaussianDistributions.jl/blob/master/src/bivariate.jl
const lnodes = [-0.9997137267734413,-0.9984919506395958,-0.9962951347331251,-0.9931249370374434,-0.9889843952429918,-0.983877540706057,-0.9778093584869183,-0.9707857757637063,-0.9628136542558156,-0.9539007829254917,-0.944055870136256,-0.9332885350430795,-0.921609298145334,-0.9090295709825297,-0.895561644970727,-0.8812186793850184,-0.8660146884971647,-0.8499645278795913,-0.8330838798884008,-0.8153892383391763,-0.7968978923903145,-0.7776279096494956,-0.7575981185197073,-0.7368280898020207,-0.7153381175730565,-0.693149199355802,-0.6702830156031411,-0.6467619085141293,-0.6226088602037078,-0.5978474702471789,-0.5725019326213813,-0.5465970120650943,-0.5201580198817632,-0.493210789208191,-0.4657816497733582,-0.43789740217203155,-0.40958529167830166,-0.38087298162462996,-0.3517885263724217,-0.32236034390052926,-0.292617188038472,-0.26258812037150336,-0.23230248184497404,-0.20178986409573646,-0.1710800805386034,-0.1402031372361141,-0.10918920358006115,-0.07806858281343654,-0.046871682421591974,-0.015628984421543188,0.015628984421543188,0.046871682421591974,0.07806858281343654,0.10918920358006115,0.1402031372361141,0.1710800805386034,0.20178986409573646,0.23230248184497404,0.26258812037150336,0.292617188038472,0.32236034390052926,0.3517885263724217,0.38087298162462996,0.40958529167830166,0.43789740217203155,0.4657816497733582,0.493210789208191,0.5201580198817632,0.5465970120650943,0.5725019326213813,0.5978474702471789,0.6226088602037078,0.6467619085141293,0.6702830156031411,0.693149199355802,0.7153381175730565,0.7368280898020207,0.7575981185197073,0.7776279096494956,0.7968978923903145,0.8153892383391763,0.8330838798884008,0.8499645278795913,0.8660146884971647,0.8812186793850184,0.895561644970727,0.9090295709825297,0.921609298145334,0.9332885350430795,0.944055870136256,0.9539007829254917,0.9628136542558156,0.9707857757637063,0.9778093584869183,0.983877540706057,0.9889843952429918,0.9931249370374434,0.9962951347331251,0.9984919506395958,0.9997137267734413,]
const lweights = [0.0007346344905056717,0.001709392653518105,0.0026839253715534818,0.003655961201326376,0.004624450063422119,0.005588428003865517,0.006546948450845323,0.007499073255464713,0.008443871469668972,0.009380419653694457,0.010307802574868971,0.011225114023185977,0.012131457662979496,0.013025947892971542,0.013907710703718773,0.014775884527441305,0.015629621077546,0.016468086176145213,0.01729046056832358,0.018095940722128112,0.018883739613374903,0.019653087494435305,0.02040323264620943,0.021133442112527635,0.021843002416247394,0.02253122025633627,0.02319742318525412,0.02384096026596821,0.024461202707957052,0.025057544481579586,0.025629402910208116,0.026176219239545672,0.02669745918357096,0.02719261344657688,0.027661198220792382,0.02810275565910117,0.028516854322395098,0.028903089601125212,0.029261084110638276,0.029590488059912642,0.029890979593332836,0.030162265105169145,0.030404079526454818,0.030616186583980444,0.03079837903115259,0.03095047885049098,0.03107233742756652,0.031163835696209907,0.031224884254849355,0.03125542345386336,0.03125542345386336,0.031224884254849355,0.031163835696209907,0.03107233742756652,0.03095047885049098,0.03079837903115259,0.030616186583980444,0.030404079526454818,0.030162265105169145,0.029890979593332836,0.029590488059912642,0.029261084110638276,0.028903089601125212,0.028516854322395098,0.02810275565910117,0.027661198220792382,0.02719261344657688,0.02669745918357096,0.026176219239545672,0.025629402910208116,0.025057544481579586,0.024461202707957052,0.02384096026596821,0.02319742318525412,0.02253122025633627,0.021843002416247394,0.021133442112527635,0.02040323264620943,0.019653087494435305,0.018883739613374903,0.018095940722128112,0.01729046056832358,0.016468086176145213,0.015629621077546,0.014775884527441305,0.013907710703718773,0.013025947892971542,0.012131457662979496,0.011225114023185977,0.010307802574868971,0.009380419653694457,0.008443871469668972,0.007499073255464713,0.006546948450845323,0.005588428003865517,0.004624450063422119,0.003655961201326376,0.0026839253715534818,0.001709392653518105,0.0007346344905056717,]

_Phi(x) = Distributions.cdf(Distributions.Normal(), x)
_phi(x) = Distributions.pdf(Distributions.Normal(), x)

# bivariate density
_phi(x, y, rho) =  1/(2*pi*sqrt(1-rho^2))*exp(-0.5*(x^2 + y^2 - 2x*y*rho)/(1-rho^2))
# transformed to the interval [-1,1]
_phigauss(s, x, y, rho) = (rho)/2 * _phi(x, y, 0.5*rho*(s + 1))
# substitute r = sqrt(1-rho^2) for backward integration
function _phiback(x, y, r)
    r̄ = sqrt(1-r^2)
    (1/(2pi*r̄))*exp(-0.5*(x^2 + y^2 - 2x*y*r̄)/r^2)
end
# transformed to the interval [-1,1]
_phibackgauss(s, x, y, rho) = 0.5*sqrt(1-rho^2)*_phiback(x, y, (sqrt(1-rho^2))*0.5*(s + 1))

function _Phi(x, y, ρ)
    if ρ == 1
        return _Phi(min(x, y))
    end

    if x == Inf || y == Inf
        _Phi(min(x, y))
    elseif x == -Inf || y == -Inf
        0.0
    elseif ρ < -0.95
        _Phi(x) - _Phi(x, -y, -ρ)
    elseif ρ > 0.95
        _Phi(min(x,y)) - sum( lweights[i] * _phibackgauss(lnodes[i], x, y, ρ) for i in 1:length(lnodes) )
    else
        _Phi(x)*_Phi(y) + sum( lweights[i] * _phigauss(lnodes[i], x, y, ρ) for i in 1:length(lnodes) )
    end
end

"""
    estimate_multivariate_distribution(firing_rates, [covariance_matrix, dt])

Given a set of neuron firing rates, construct a multivariate Gaussian distribution
where sampling randomly from the distribution (.>=0) returns a series of poisson
spiketrains with the associated covariance structure (defaults to identity).

This code is based on the following reference:
Generating Spike Trains with Specified Correlation Coefficients, Macke et al., (2009)

Reference code is implemented here:
https://github.com/mackelab/dg_python
"""
function estimate_multivariate_distribution(firing_rates::AbstractVector{<:Real}, covariance_matrix::AbstractMatrix{<:Real}=LinearAlgebra.I(length(firing_rates)), dt::Real=0.001; ignore_warnings::Bool=false)
    @assert length(firing_rates) == size(covariance_matrix, 1) == size(covariance_matrix, 2)
    simulated_gammas = zeros(size(covariance_matrix, 1))
    simulated_covariance = ones(size(covariance_matrix))
    function is_positive_semi_definite(m::AbstractMatrix)
        try 
            LinearAlgebra.cholesky(m)
        catch
            return false
        end
        return true
    end
    rs = firing_rates .* dt # Convert to probability
    simulated_gammas = [Distributions.quantile(Distributions.Normal(), r) for r in rs]

    for i = 1:length(simulated_gammas)
        for j = i+1:length(simulated_gammas)
            fxn = lambda -> _Phi(simulated_gammas[i], simulated_gammas[j], lambda) - rs[i] * rs[j] - covariance_matrix[i, j]
            try
                simulated_covariance[i, j] = MovementToolbox.find_root_in_bounds(fxn, (-1, 1), verbose=false)
            catch e
                simulated_covariance[i, j] = 0.0
                @warn e maxlog=1
            end
            simulated_covariance[j, i] = simulated_covariance[i, j]
        end
    end


    if ! is_positive_semi_definite(simulated_covariance)
        ignore_warnings == false && @warn "Simulated matrix is not positive semi-definite"
        simulated_covariance = higham_correction(simulated_covariance)
    end

    return Distributions.MultivariateNormal(simulated_gammas, simulated_covariance)
end

"""
    higham_correction(m)

Converts an input symmetric matrix into a positive semi-definite matrix
using the Higham iterative projection to minimize the Forbenius norm between
A and M

References: 
NJ Higham, Computing the nearest correlation matrix - a problem from finance, IMA Journal of
Numerical Analysis, 2002

Implementation based on :
https://nhigham.com/2013/02/13/the-nearest-correlation-matrix/
Via:
https://github.com/mackelab/dg_python/blob/b21193dc3ce86167d6a552487f87f881f695f6cf/dg_python/dichot_gauss.py#L59
"""
function higham_correction(m::AbstractMatrix{<:Real}; max_iterations::Real=1e5, tol::Real=1e-10)
    function positive_projection(m::AbstractMatrix{<:Real})
        vals = LinearAlgebra.eigvals(m)
        vecs = LinearAlgebra.eigvecs(m)
 
        vals[real.(vals) .< 0.0] .= 1e-12
        A = vecs * LinearAlgebra.diagm(vals) * vecs'
        A .= (A .+ A') ./ 2.0 # Make symmetric
        @assert all(LinearAlgebra.eigvals(A) .>= 0.0)
        return A
    end

    function identity_projection(m::AbstractMatrix{<:Real})
        u = LinearAlgebra.diagm(LinearAlgebra.diag(m - LinearAlgebra.I(size(m, 1))))
        return m - u
    end

    iterations = 0
    DS = 0.0
    Y_0 = deepcopy(m)
    Y_n = nothing
    X_0 = deepcopy(m)
    X_n = nothing
    delta = Inf
    while (iterations < max_iterations) && (delta > tol)
        R = Y_0 .- DS
        X_n = positive_projection(R)
        DS = X_n .- R
        Y_n = identity_projection(X_n)
        
        delta_x = LinearAlgebra.norm(X_n .- X_0) ./ LinearAlgebra.norm(X_0)
        delta_y = LinearAlgebra.norm(Y_n .- Y_0) ./ LinearAlgebra.norm(Y_0)
        delta_xy = LinearAlgebra.norm(Y_n .- X_n) ./ LinearAlgebra.norm(Y_0)
        delta = max(delta_x, delta_y, delta_xy)
        X_0 = X_n
        Y_0 = Y_n
        iterations = iterations + 1
    end
    if iterations >= max_iterations
        error("Maximum number of iterations reached.")
    end
    vals = LinearAlgebra.eigvals(Y_n) 
    if any(vals .< tol)
        Y_n = positive_projection(m)
    end
    return Y_n
end

"""
    generate_spiketrain_from_multivariate_distribution(mv; [dt, duration])

Given a N-dimensional multivariate distribution, pull random (correlated) numbers from the distribution.
Values greater than 0 and converted into spikes (binary spike train). These binary events correspond
to the firing of the N-th neuron.
"""
function generate_spiketrain_from_multivariate_distribution(mv::Distributions.FullNormal; dt::Real=1e-3, duration::Real=60.0, absolute_refractory_period::Real=2e-3)
    # Generate our spiketrains
    num_timepoints = Integer(ceil(duration / dt))
    num_neurons = length(mv.μ)
    absolute_refractory_period_indices = Integer(ceil(absolute_refractory_period / dt))
    spiketrains = falses(num_neurons, num_timepoints)
    last_spike = ones(num_neurons) * -Inf
    for i = 1:num_timepoints
        potential_spikes = rand(mv) .>= 0
        for j = 1:num_neurons
            if potential_spikes[j] == true && i - last_spike[j] > absolute_refractory_period_indices
                last_spike[j] = i
                spiketrains[j, i] = true
            end 
        end
    end
    return spiketrains
end
