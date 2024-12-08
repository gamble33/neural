function neural_net(input::Vector{Float32}, weights::Matrix{Float32})
    weights * input
end

function error(actual::Float32, expected::Float32)
    (actual - expected)^2
end

avg_games = [8.5, 9.5, 9.9, 9.0]
win_percentage = [0.65, 0.8, 0.8, 0.9]
n_fans = [1.2, 1.3, 0.5, 1.0]
index = 0

input::Vector{Float32} = [0.5]
weights::Matrix{Float32} = reshape([0.5], 1, 1) # 1x1 Matrix

# Learning Process
step::Float32 = 0.001
for i in 0:1101
    target_pred::Float32 = 0.8

    up_pred = neural_net(input, weights .+ step)
    up_err = error(up_pred[1], target_pred)

    down_pred = neural_net(input, weights .- step)
    down_err = error(down_pred[1], target_pred)

    if down_err < up_err
        weights .-= step
    else
        weights .+= step
    end
end


print(neural_net(input, weights))