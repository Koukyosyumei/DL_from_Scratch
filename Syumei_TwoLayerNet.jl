module Syumei_TwoLayerNet

include("./Syumei_Functions.jl")
include("./Syumei_BasicLayer.jl")
using ..Syumei_Functions
using ..Syumei_BasicLayer

using LinearAlgebra
using Statistics

export TwoLayerNet, predict, loss, accuracy, TwoLayerNetGrads, gradient_syumei, numerical_gradient, applygradient!

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

function gradient_syumei(net::TwoLayerNet{T}, x::AbstractArray{T}, t::AbstractArray{T}) where T<:Any
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

end
