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
