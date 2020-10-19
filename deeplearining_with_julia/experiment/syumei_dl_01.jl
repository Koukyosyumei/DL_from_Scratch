
using Gadfly
using BenchmarkTools
using Flux, Flux.Data.MNIST
using JLD
using Images
using LinearAlgebra
using Statistics

function onehot(t::AbstractVector, l::AbstractVector)
    r = zeros(Float64, length(l), length(t))
    for i = 1:length(t)
        r[findfirst(x->x==t[i], l), i] = 1
    end
    r
end

abstract type AbstractLayer end

mutable struct MulLayer{T} <: AbstractLayer
    x::T
    y::T
end

MulLayer() = MulLayer(zero(Float64), zero(Float64))

function forward(lyr::MulLayer{T}, x::T, y::T) where T<:Any
    lyr.x = x
    lyr.y = y
    x * y
end

@inline forward(lyr::MulLayer{T}, x, y) where T<:Any = forward(lyr, T(x), T(y))

function backward(lyr::MulLayer{T}, dout::T) where T<:Any
    dx = dout * lyr.y
    dy = dout * lyr.x
    (dx, dy)
end

mutable struct AddLayer <: AbstractLayer end

function forward(lyr::AddLayer, x, y)
    x + y
end

function backward(lyr::AddLayer, dout)
    (dout, dout)
end

mutable struct ReluLayer <: AbstractLayer
    mask::AbstractArray{Bool}
    ReluLayer() = new()
end

function forward(lyr::ReluLayer, x::AbstractArray{T}) where T<:Any
    lyr.mask = (x .<= 0)
    mask = (x .<= 0)
    #mask = lyr.mask
    out = copy(x)
    out[mask] .= zero(T)
    out
end

function backward(lyr::ReluLayer, dout::AbstractArray{T}) where T<:Any
    dout[lyr.mask] .= zero(T)
    dout
end

mutable struct SigmoidLayer{T} <: AbstractLayer
    out::T
end
SigmoidLayer() = SigmoidLayer(zero(Float64))

function forward(lyr::SigmoidLayer{T}, x::T) where T<:Any
    lyr.out = 1 ./ (1 .+ exp(-x))
end

function backward(lyr::SigmoidLayer{T}, dout::T) where T<:Any
    dout .* (1 .- lyr.out) .* lyr.out
end

mutable struct AffineLayer{T} <: AbstractLayer

    W::AbstractMatrix{T}
    b::AbstractVector{T}
    x::AbstractArray{T}
    dW::AbstractMatrix{T}
    db::AbstractVector{T}

    function AffineLayer(W::AbstractMatrix{T}, b::AbstractVector{T}) where T<:Any
        lyr = new{T}()
        lyr.W = W
        lyr.b = b
        lyr
    end

end

function forward(lyr::AffineLayer{T}, x::AbstractArray{T}) where T<:Any
    lyr.x = x
    lyr.W * x .+ lyr.b
end

function backward(lyr::AffineLayer{T}, dout::AbstractArray{T}) where T <:Any
    dx = lyr.W' * dout
    lyr.dW = dout * lyr.x'
    lyr.db = _sumvec(dout)
    dx
end
@inline _sumvec(dout::Vector) = dout
@inline _sumvec(dout::Matrix) = vec(mapslices(sum, dout, dims=2))
@inline _sumvec(dout::Array) = vec(mapslices(sum, dout, dims=2))

function softmax(a::AbstractVector{T}) where T<:Real
    c = maximum(a)  # オーバーフロー対策
    exp_a = exp.(a .- c)
    exp_a ./ sum(exp_a)
end

function softmax(a::AbstractMatrix{T}) where T<:Real
    mapslices(softmax, a, dims=1)
end

function crossentropyerror(y::Vector, t::Vector)
    δ = 1e-7  # アンダーフロー対策
    # -sum(t .* log(y .+ δ))
    -(t ⋅ log(y .+ δ))
end

function crossentropyerror(y::Array, t::Array)
    batch_size = size(y, 2)
    δ = 1e-7  # アンダーフロー対策
    # -sum(t .* log(y .+ δ)) / batch_size
    -dot(t, log.(y .+ δ)) / batch_size
end

function crossentropyerror(y::Matrix, t::Matrix)
    batch_size = size(y, 2)
    δ = 1e-7  # アンダーフロー対策
    # -sum(t .* log(y .+ δ)) / batch_size
    -dot(t, log.(y .+ δ)) / batch_size
end

function crossentropyerror(y::Matrix, t::Vector)
    batch_size = size(y, 2)
    δ = 1e-7  # アンダーフロー対策
    -sum([log.(y[t[i]+1, i]) for i=1:batch_size] .+ δ) / batch_size
end

mutable struct SoftmaxWithLossLayer!{T} <: AbstractLayer
    loss::T
    y::AbstractArray{T}
    t::AbstractArray{T}
    #(::Type{SoftmaxWithLossLayer{T}}){T}() = new{T}
    SoftmaxWithLossLaye12r() = new{T}() where T<:Any
end

mutable struct SoftmaxWithLossLayer<: AbstractLayer
    loss::Float64
    y::AbstractArray{Float64}
    t::AbstractArray{Float64}
    #(::Type{SoftmaxWithLossLayer{T}}){T}() = new{T}
    SoftmaxWithLossLayer() = new()
end

function forward(lyr::SoftmaxWithLossLayer, x::AbstractArray{T}, t::AbstractArray{T}) where T<:Any
    lyr.t = t
    y = lyr.y = softmax(x)
    lyr.loss = crossentropyerror(y, t)
end

function backward(lyr::SoftmaxWithLossLayer, dout::T=1) where T<:Any
    dout .* _swlvec(lyr.y, lyr.t)
end
@inline _swlvec(y::Array, t::Vector) = y .- t
@inline _swlvec(y::Array, t::Matrix) = (y .- t) / size(t)[2]

mutable struct TwoLayerNet{T}
    a1lyr::AffineLayer{T}
    relu1lyr::ReluLayer
    a2lyr::AffineLayer{T}
    softmaxlyr::SoftmaxWithLossLayer
end

function TwoLayerNet{T}(input_size::Int, hidden_size::Int, output_size::Int,
        weight_init_std::Float64=0.01)::TwoLayerNet{T} where T
    W1 = weight_init_std .* randn(T, hidden_size, input_size)
    b1 = zeros(T, hidden_size)
    W2 = weight_init_std .* randn(T, output_size, hidden_size)
    b2 = zeros(T, output_size)
    a1lyr = AffineLayer(W1, b1)
    relu1lyr = ReluLayer()
    a2lyr = AffineLayer(W2, b2)
    softmaxlyr = SoftmaxWithLossLayer()
    # TwoLayerNet(W1, b1, W2, b2, a1lyr, relu1lyr, a2lyr, softmaxlyr)
    TwoLayerNet(a1lyr, relu1lyr, a2lyr, softmaxlyr)
end

function predict(net::TwoLayerNet{T}, x::AbstractArray{T}) where T<:Any
    a1 = forward(net.a1lyr, x)
    z1 = forward(net.relu1lyr, a1)
    a2 = forward(net.a2lyr, z1)
    # softmax(a2)
    a2
end


function loss(net::TwoLayerNet{T}, x::AbstractArray{T}, t::AbstractArray{T}) where T<:Any
    y = predict(net, x)
    forward(net.softmaxlyr, y, t)
end

function accuracy(net::TwoLayerNet{T}, x::AbstractArray{T}, t::AbstractArray{T}) where T<:Any
    y = vec(mapslices(argmax, predict(net, x), dims=1))
    if ndims(t) > 1 t = vec(mapslices(argmax, t, dims=1)) end
    mean(y .== t)
end

struct TwoLayerNetGrads{T}
    W1::AbstractMatrix{T}
    b1::AbstractVector{T}
    W2::AbstractMatrix{T}
    b2::AbstractVector{T}
end

function gradient4tln(net::TwoLayerNet{T}, x::AbstractArray{T}, t::AbstractArray{T}) where T<:Any
    # forward
    loss(net, x, t)
    # backward
    dout = one(T)
    dz2 = backward(net.softmaxlyr, dout)
    da2 = backward(net.a2lyr, dz2)
    dz1 = backward(net.relu1lyr, da2)
    da1 = backward(net.a1lyr, dz1)
    TwoLayerNetGrads(net.a1lyr.dW, net.a1lyr.db, net.a2lyr.dW, net.a2lyr.db)
end

function numerical_gradient(f, x::Vector)
    h = 1e-4 # 0.0001
    # (f(x+h) - f(x-h)) / 2h
    map(1:length(x)) do idx
        tmp_val = x[idx]
        # f(x+h)
        x[idx] += h
        fxh1 = f(x)
        # f(x-h)
        x[idx] -= 2h
        fxh2 = f(x)
        # restore
        x[idx] = tmp_val
        (fxh1 - fxh2) / 2h
    end
end
function numerical_gradient(f, x::AbstractArray{T,N}) where T <:Any where N<:Any
    h = 1e-4 # 0.0001
    # (f(x+h) - f(x-h)) / 2h
    reshape(map(1:length(x)) do idx
        tmp_val = x[idx]
        # f(x+h)
        x[idx] += h
        fxh1 = f(x)
        # f(x-h)
        x[idx] -= 2h
        fxh2 = f(x)
        # restore
        x[idx] = tmp_val
        (fxh1 - fxh2) / 2h
    end, size(x))
end

function numerical_gradient(net::TwoLayerNet{T}, x::AbstractArray{T}, t::AbstractArray{T}) where T<:AbstractFloat
    # W1
    dW1 = numerical_gradient(copy(net.a1lyr.W)) do W
        loss(TwoLayerNet(AffineLayer(W, net.a1lyr.b), net.relu1lyr, net.a2lyr, net.softmaxlyr), x, t)
    end
    # b1
    db1 = numerical_gradient(copy(net.a1lyr.b)) do b
        loss(TwoLayerNet(AffineLayer(net.a1lyr.W, b), net.relu1lyr, net.a2lyr, net.softmaxlyr), x, t)
    end
    # W2
    dW2 = numerical_gradient(copy(net.a2lyr.W)) do W
        loss(TwoLayerNet(net.a1lyr, net.relu1lyr, AffineLayer(W, net.a2lyr.b), net.softmaxlyr), x, t)
    end
    # b2
    db2 = numerical_gradient(copy(net.a2lyr.b)) do b
        loss(TwoLayerNet(net.a1lyr, net.relu1lyr, AffineLayer(net.a2lyr.W, b), net.softmaxlyr), x, t)
    end
    TwoLayerNetGrads(dW1, db1, dW2, db2)
end

function applygradient!(net::TwoLayerNet{T}, grads::TwoLayerNetGrads{T}, learning_rate::T) where T<:Any
    net.a1lyr.W -= learning_rate .* grads.W1
    net.a1lyr.b -= learning_rate .* grads.b1
    net.a2lyr.W -= learning_rate .* grads.W2
    net.a2lyr.b -= learning_rate .* grads.b2
end

imgs = MNIST.images()
fpt_imgs = float.(imgs)
unraveled_fpt_imgs = reshape.(fpt_imgs, :);
typeof(unraveled_fpt_imgs)

X = hcat(unraveled_fpt_imgs...)

labels = MNIST.labels()

onehot(labels, 0:9)

x_batch = X[:, 1:3]

t_batch = onehot(labels[1:3], 0:9)

y = onehot(labels, 0:9)

typeof(TwoLayerNet{Float64})

network = TwoLayerNet{Float64}(784, 50, 10)

predict(network, x_batch)

loss(network, x_batch, t_batch)

grad_numerical = numerical_gradient(network, x_batch, t_batch)

grad_backprop = gradient4tln(network, x_batch, t_batch)

extrema(grad_numerical.W1)

extrema(grad_backprop.W1)

extrema(grad_numerical.b1)

extrema(grad_backprop.b1)

extrema(grad_numerical.W2)

extrema(grad_backprop.W2)

extrema(grad_numerical.b2)

extrema(grad_backprop.b2)

network = TwoLayerNet{Float64}(784, 50, 10)

x_train = X[:, 1:50000]
x_test  = X[:, 50001:60000]

y_train = y[:, 1:50000]
y_test  = y[:, 50001:60000]

iters_num = 10000;
train_size = size(x_train, 2); # => 60000
batch_size = 100;
learning_rate = Float32(0.1);

train_size

train_loss_list = Float64[];
train_acc_list = Float64[];
test_acc_list = Float64[];

iter_per_epoch = max(train_size ÷ batch_size, 1)



for i = 1:iters_num
    batch_mask = rand(1:train_size, batch_size)
    x_batch = x_train[:, batch_mask]
    t_batch = y_train[:, batch_mask]

    # 誤差逆伝播法によって勾配を求める
    grads = gradient4tln(network, x_batch, t_batch)

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

plot(x=1:length(train_loss_list), y=train_loss_list, Geom.line)

xs = [1:length(train_acc_list);]
y1 = train_acc_list
y2 = test_acc_list
c1 = repeat(["train_acc"], length(xs))
c2 = repeat(["test_acc"], length(xs))
plot(x=[xs;xs], y=[y1;y2], color=[c1;c2], Geom.line)
