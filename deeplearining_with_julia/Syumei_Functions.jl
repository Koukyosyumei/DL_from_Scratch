module Syumei_Functions

using Gadfly
using BenchmarkTools
using Flux, Flux.Data.MNIST
using JLD
using Images
using LinearAlgebra
using Statistics

export onehot, softmax_syumei, crossentropyerror

function onehot(t::AbstractVector, l::AbstractVector)
    """     Summary line

        one_hot エンコーディングをする

        Args
                t : 変換したいベクトル
                l : tに含まれるカテゴリー変数のリスト

        Returns
                r : 変換されたベクトル

        examples
                imgs = MNIST.images()
                fpt_imgs = float.(imgs)
                unraveled_fpt_imgs = reshape.(fpt_imgs, :);
                typeof(unraveled_fpt_imgs)
                X = hcat(unraveled_fpt_imgs...)
                labels = MNIST.labels()
                y = Syumei_Functions.onehot(labels, 0:9)

    """
    r = zeros(Float64, length(l), length(t))
    for i = 1:length(t)
        r[findfirst(x->x==t[i], l), i] = 1
    end
    r
end

function softmax_syumei(a::AbstractVector{T}) where T<:Real
    """             Summary line

        ソフトマックス関数

        Args
                a     : ベクトル ・　中身はIntでもFloatでも可
        Returns
                exp_a :
    """
    c = maximum(a)  # オーバーフロー対策
    exp_a = exp.(a .- c)
    exp_a ./ sum(exp_a)
end

function softmax_syumei(a::AbstractMatrix{T}) where T<:Real
    """   Summaryline

        softmax_syumei を行列にも適用できるように拡張

    """
    mapslices(softmax_syumei, a, dims=1)
end

# 以下、crossentropyerror の定義
# ベクトル・配列・行列すべてに適用できるようにしている

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

end
