using System.Collections;
using System.Collections.Generic;
using System;
using System.Linq;
using UnityEngine;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using System.Runtime.ConstrainedExecution;

using number = System.Double;


[System.Serializable]
public struct NnActivations<T> : IDisposable
    where T:struct
{
    public NativeArray<T> activations;

    public int lengthOfNodes => this.activations.Length - 1;
    public int lengthWithBias => this.activations.Length;

    public T this[int i]
    {
        get => this.activations[i];
        set => this.activations[i] = value;
    }

    public void Dispose()
    {
        if (!this.activations.IsCreated) return;

        this.activations.Dispose();
    }
}


[System.Serializable]
public struct NnWeights<T> : IDisposable
    where T:struct
{
    public NativeArray<T> weights;
    public int width;

    public int widthOfNodes => this.width - 1;
    public int widthWithBias => this.width;

    public int length => this.weights.Length;

    public T this[int ix, int iy]
    {
        get => this.weights[iy * this.width + ix];
        set => this.weights[iy * this.width + ix] = value;
    }

    public void Dispose()
    {
        if (!this.weights.IsCreated) return;

        this.weights.Dispose();
    }
}



public interface IActivationFunction
{
    number Activate(number u);
    number Prime(number a);
    void InitWeights(NnWeights<number> weights);
}

public struct ReLU : IActivationFunction
{
    public number Activate(number u) => max(u, 0.0);
    public number Prime(number a) => sign(a);
    public void InitWeights(NnWeights<number> weights) => weights.InitHe();
}
public struct Sigmoid : IActivationFunction
{
    public number Activate(number u) => 1.0 / (1.0 + exp(-u));
    //public number Activate(number u)
    //{
    //    var x = (1.0 + exp(-u));
    //    if (x == 0) return 0;
    //    return 1.0 / x;
    //}
    public number Prime(number a) => a * (1.0 - a);
    public void InitWeights(NnWeights<number> weights) => weights.InitXivier();
}
public struct Affine : IActivationFunction
{
    public number Activate(number u) => u;
    public number Prime(number a) => 1;
    public void InitWeights(NnWeights<number> weights) => weights.InitRandom();
}

