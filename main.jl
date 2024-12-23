const Vec = Vector{Float32}
const Mat = Matrix{Float32}

function neural_net(input::Vec, weights::Mat)::Vec
    weights * input
end

"""
The loss (error) functoin calculates the square of the difference between the expected and actual 
values determined by a neural network.

# Arguments
- `y`: The vector of the actual values determined by the neural net.
- `ŷ`: The vector of the expected values.

# Returns
A vector containing the error in each output node.
"""
function error(y::Vec, ŷ::Vec)::Vec
    (y - ŷ) .^ 2
end

avg_games = [8.5, 9.5, 9.9, 9.0]
win_percentage = [0.65, 0.8, 0.8, 0.9]
n_fans = [1.2, 1.3, 0.5, 1.0]
index = 0

input::Vec = [avg_games[1], win_percentage[1], n_fans[1]]
target_pred::Vec = [0.1, 1.0, 0.1]

function learn()
    weights::Mat = [
        0.1  0.1 -0.3
        0.1  0.2  0.0
        0.0  1.3  0.1
    ]

    α = 0.01

    # Learning Process
    for _ in 0:5
        pred = neural_net(input, weights)
        err = error(pred, target_pred)
        Δ = pred - target_pred
        Δw = Δ * input' # Outer product of Δ vector and transposed input vector.
        weights -= Δw * α

        println(Δw)
        println()
        println()
    end


    println(neural_net(input, weights))
end

learn()