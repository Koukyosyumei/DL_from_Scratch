module Syumei_Train

include("./Syumei_Functions.jl")
include("./Syumei_BasicLayer.jl")
include("./Syumei_TwoLayerNet.jl")
using ..Syumei_BasicLayer
using ..Syumei_Functions
using ..Syumei_TwoLayerNet

export train

function train(network, x_train, y_train, x_test, y_test, iters_num, batch_size, learning_rate)
    train_size = size(x_train, 2)

    train_loss_list = Float64[];
    train_acc_list = Float64[];
    test_acc_list = Float64[];

    iter_per_epoch = max(train_size ÷ batch_size, 1)

    for i = 1:iters_num
        batch_mask = rand(1:train_size, batch_size)
        x_batch = x_train[:, batch_mask]
        t_batch = y_train[:, batch_mask]

        # 誤差逆伝播法によって勾配を求める
        grads = gradient_syumei(network, x_batch, t_batch)

        # 更新
        learning_rate = 0.01::Float64
        applygradient!(network, grads, learning_rate)

        _loss = loss(network, x_batch, t_batch)
        push!(train_loss_list, _loss)

        if i % iter_per_epoch == 1
            train_acc = accuracy(network, x_train, y_train)
            test_acc = accuracy(network, x_test, y_test)
            push!(train_acc_list, train_acc)
            push!(test_acc_list, test_acc)
            println("$(i-1): train_acc=$(train_acc) / test_acc=$(test_acc)")
        end
    end

    final_train_acc = accuracy(network, x_train, y_train)
    final_test_acc = accuracy(network, x_test, y_test)
    push!(train_acc_list, final_train_acc)
    push!(test_acc_list, final_test_acc)
    println("final: train_acc=$(final_train_acc) / test_acc=$(final_test_acc)")

    train_acc_list, test_acc_list, train_loss_list, network
end

end
