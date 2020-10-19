include("./Syumei_Functions.jl")
include("./Syumei_BasicLayer.jl")
include("./Syumei_TwoLayerNet.jl")
include("./Syumei_Train.jl")
using ..Syumei_Functions
using ..Syumei_BasicLayer
using ..Syumei_TwoLayerNet
using ..Syumei_Train

using Gadfly
using BenchmarkTools
using Flux, Flux.Data.MNIST
using JLD
using Images
using LinearAlgebra
using Statistics
using Dates

function main()

    #-----------------------------
    # MNISTのデータの取得
    imgs = MNIST.images()
    fpt_imgs = float.(imgs)
    unraveled_fpt_imgs = reshape.(fpt_imgs, :);
    typeof(unraveled_fpt_imgs)
    X = hcat(unraveled_fpt_imgs...)
    labels = MNIST.labels()
    y = Syumei_Functions.onehot(labels, 0:9)

    x_train = X[:, 1:50000]
    x_test  = X[:, 50001:60000]

    y_train = y[:, 1:50000]
    y_test  = y[:, 50001:60000]

    #-----------------------------------------------------------------------------
    # モデルの定義
    network = Syumei_TwoLayerNet.TwoLayerNet{Float64}(784, 50, 10)

    #-------------------------------------------------------------
    # 学習

    iters_num = 10000
    batch_size = 100
    learning_rate = Float32(0.1)

    train_acc_list, test_acc_list, train_loss_list, network = train(network, x_train, y_train,
                                                   x_test, y_test, iters_num,batch_size, learning_rate)

    #---------------------------------------------------------------
    # 結果の可視化
    p = plot(x=1:length(train_loss_list), y=train_loss_list, Geom.line)
    path = Dates.format(now(), "yyyy-mm-dd HH:MM:SS") * ".svg"
    img = SVG(path, 14cm, 8cm)
    draw(img, p)

    xs = [1:length(train_acc_list);]
    y1 = train_acc_list
    y2 = test_acc_list
    c1 = repeat(["train_acc"], length(xs))
    c2 = repeat(["test_acc"], length(xs))
    p = plot(x=[xs;xs], y=[y1;y2], color=[c1;c2], Geom.line)
    path = Dates.format(now(), "yyyy-mm-dd HH:MM:SS") * ".svg"
    img = SVG(path, 14cm, 8cm)
    draw(img, p)

end

main()
