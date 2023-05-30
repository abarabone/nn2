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

using number = System.Single;
using number4 = Unity.Mathematics.float4;


[System.Serializable]
public struct NnActivations<T> : IDisposable
    where T:struct
{
    public NativeArray<T> activations;

    public int lengthOfNodes => this.activations.Length - 1;
    public int lengthWithBias => this.activations.Length;
    public int length => this.activations.Length;

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


//struct aaa
//{
//    void aaaaa() => new NnActivations4<NnActivation4<number4>, number4>();
//}

[System.Serializable]
public struct NnActivations4 : IDisposable
{
    public NativeArray<number4> currents;

    public int lengthOfNodes => this.currents.Length * 4;
    public int lengthOfUnits => this.currents.Length;

    //public IActivationValue<number4> this[int i]
    //{
    //    get => new NnActivation4() { value = this.currents[i] };
    //    set => this.currents[i] = value.Value;
    //}
    public number4 this[int i]
    {
        get => this.currents[i];
        set => this.currents[i] = value;
    }

    public void Dispose()
    {
        if (!this.currents.IsCreated) return;

        this.currents.Dispose();
    }
}
//public interface IActivationValue<T> where T : struct
//{
//    public T Value { get; set; }
//    //public void init(NativeArray<T> na, int i);
//}
//public struct NnActivation4 : IActivationValue<number4>
//{
//    public number4 value;
//    public number4 Value
//    {
//        get => this.value;
//        set => this.value = value;
//    }
//    //public void init(NativeArray<T> na, int i)
//    //{
//    //    this.currents = na;
//    //}

//    static public number4 operator *(NnActivation4 l, NnWeights4 r)
//    {
//        return 0;
//    }
//}


[System.Serializable]
public struct NnWeights4 : IDisposable
{
    public NativeArray<number4> cxp;
    public int width;

    public int widthOfNodes => this.width * 4;
    public int widthOfUnits => this.width;

    public int length => this.cxp.Length;

    public number4 this[int ic, int ip]
    {
        get => this.cxp[ip * this.width + ic];
        set => this.cxp[ip * this.width + ic] = value;
    }

    public void Dispose()
    {
        if (!this.cxp.IsCreated) return;

        this.cxp.Dispose();
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
    public number Activate(number u) => max(u, 0);
    public number Prime(number a) => sign(a);
    public void InitWeights(NnWeights<number> weights) => weights.InitHe();
}
public struct Sigmoid : IActivationFunction
{
    public number Activate(number u) => 1 / (1 + exp(-u));
    public number Prime(number a) => a * (1 - a);
    public void InitWeights(NnWeights<number> weights) => weights.InitXivier();
}
public struct Affine : IActivationFunction
{
    public number Activate(number u) => u;
    public number Prime(number a) => 1;
    public void InitWeights(NnWeights<number> weights) => weights.InitRandom();
}

public interface IActivationFunction4
{
    number4 Activate(number4 u);
    number4 Prime(number4 a);
    void InitWeights(NnWeights4 weights);
}
public struct ReLU4 : IActivationFunction4
{
    public number4 Activate(number4 u) => max(u, 0);
    public number4 Prime(number4 a) => sign(a);
    public void InitWeights(NnWeights4 weights) { }// => weights.InitHe();
}
public struct Sigmoid4 : IActivationFunction4
{
    public number4 Activate(number4 u) => 1 / (1 + exp(-u));
    public number4 Prime(number4 a) => a * (1 - a);
    public void InitWeights(NnWeights4 weights) { }// => weights.InitXivier();
}
public struct Affine4 : IActivationFunction4
{
    public number4 Activate(number4 u) => u;
    public number4 Prime(number4 a) => 1;
    public void InitWeights(NnWeights4 weights) { }// => weights.InitRandom();
}