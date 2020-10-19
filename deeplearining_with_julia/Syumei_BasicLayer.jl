module Syumei_BasicLayer

using Gadfly
using BenchmarkTools
using Flux, Flux.Data.MNIST
using JLD
using Images
using LinearAlgebra
using Statistics

include("./Syumei_Functions.jl")
using ..Syumei_Functions

export AbstractLayer, MulLayer, forward, backward, AddLayer, ReluLayer, SigmoidLayer, AffineLayer, SoftmaxWithLossLayer

#-------------------------------------------------------------------------------
#Layer の大本になる型を定義 (Pythonでいう抽象クラス的な)
#この型を基に、様々なレイヤーを定義していきたい
abstract type AbstractLayer end

#-------------------------------------------------------------------------------
# 二つの引数 xとy を掛け合わせるレイヤー
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

#-------------------------------------------------------------------------------
# 二つの引数 xとy を足し合わせるレイヤー
mutable struct AddLayer <: AbstractLayer end

function forward(lyr::AddLayer, x, y)
    x + y
end

function backward(lyr::AddLayer, dout)
    (dout, dout)
end

#-------------------------------------------------------------------------------
# ReluLayer の実装
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

#-------------------------------------------------------------------------------
# SigmoidLayerの実装
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

#-------------------------------------------------------------------------------
# 結合層の実装
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
#-------------------------------------------------------------------------------
# Softmax を適用するレイヤーの実装
mutable struct SoftmaxWithLossLayer<: AbstractLayer
    loss::Float64
    y::AbstractArray{Float64}
    t::AbstractArray{Float64}
    #(::Type{SoftmaxWithLossLayer{T}}){T}() = new{T}
    SoftmaxWithLossLayer() = new()
end

function forward(lyr::SoftmaxWithLossLayer, x::AbstractArray{T}, t::AbstractArray{T}) where T<:Any
    lyr.t = t
    y = lyr.y = softmax_syumei(x)
    lyr.loss = crossentropyerror(y, t)
end

function backward(lyr::SoftmaxWithLossLayer, dout::T=1) where T<:Any
    dout .* _swlvec(lyr.y, lyr.t)
end
@inline _swlvec(y::Array, t::Vector) = y .- t
@inline _swlvec(y::Array, t::Matrix) = (y .- t) / size(t)[2]

end
