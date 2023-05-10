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
public struct NnLayers : IDisposable
{
    public NnLayer[] layers;

    //public NnLayers(params int[] nodeLengthList) : this(nodeLengthList)
    //{ }
    public NnLayers(int[] nodeLengthList)
    {
        this.layers = nodeLengthList
            .Prepend(0)
            .SkipLast(1)
            .Zip(nodeLengthList, (pre, cur) => new NnLayer(cur, pre))
            .ToArray()
            ;

        foreach (var l in this.layers)
        {
            l.weights.InitHe();
        }
    }

    public void Dispose()
    {
        foreach (var l in this.layers)
        {
            l.Dispose();
        }
    }

    //public void ExecuteForward3()
    //{
    //    for (var i = 1; i < this.layers.Length; i++)
    //    {
    //        var prev = this.layers[i - 1];
    //        var curr = this.layers[i];

    //        curr.Execute<ReLU>(prev);
    //    }
    //}
    //public void ExecuteBack3(NativeArray<float> corrects)
    //{
    //    {
    //        var prev = this.layers[this.layers.Length - 2];
    //        var curr = this.layers[this.layers.Length - 1];

    //        curr.ExecuteBackLast<ReLU>(prev, corrects);
    //    }

    //    for (var i = layers.Length - 1; i-- > 1;)
    //    {
    //        var prev = this.layers[i - 1];
    //        var curr = this.layers[i];
    //        var next = this.layers[i + 1];

    //        curr.ExecuteBack<ReLU>(prev, next);
    //    }
    //}

    public void ExecuteForward()
    {
        for (var i = 1; i < this.layers.Length - 1; i++)
        {
            var prev = this.layers[i - 1];
            var curr = this.layers[i];

            (prev, curr).Execute<ReLU>();
        }
        {
            var prev = this.layers[this.layers.Length - 2];
            var curr = this.layers[this.layers.Length - 1];

            (prev, curr).Execute<Sigmoid>();
        }
    }
    public void ExecuteBack(NativeArray<number> corrects, float learingRate)
    {
        {
            var prev = this.layers[this.layers.Length - 2];
            var curr = this.layers[this.layers.Length - 1];

            (prev, curr).ExecuteBackLast<Sigmoid>(corrects, learingRate);
            //Debug.Log($"back last {this.layers.Length - 1}");
        }

        for (var i = layers.Length - 1; i-- > 1;)
        {
            var prev = this.layers[i - 1];
            var curr = this.layers[i];
            var next = this.layers[i + 1];

            (prev, curr, next).ExecuteBack<ReLU>(learingRate);
            //Debug.Log($"back {i}");
        }
    }

    public void ExecuteForward2()
    {
        var dep = default(JobHandle);
        for (var i = 1; i < this.layers.Length - 1; i++)
        {
            var prev = this.layers[i - 1];
            var curr = this.layers[i];

            dep = (prev, curr).Execute2<ReLU>(dep);
        }
        {
            var prev = this.layers[this.layers.Length - 2];
            var curr = this.layers[this.layers.Length - 1];

            dep = (prev, curr).Execute2<Sigmoid>(dep);
        }

        dep.Complete();
    }
    public void ExecuteBack2(NativeArray<number> corrects, float learingRate)
    {
        var dep = default(JobHandle);

        {
            var prev = this.layers[this.layers.Length - 2];
            var curr = this.layers[this.layers.Length - 1];

            dep = (prev, curr).ExecuteBackLast2<Sigmoid>(corrects, learingRate, dep);
        }

        for (var i = layers.Length - 1; i-- > 1;)
        {
            var prev = this.layers[i - 1];
            var curr = this.layers[i];
            var next = this.layers[i + 1];

            dep = (prev, curr, next).ExecuteBack2<ReLU>(learingRate, dep);
        }

        dep.Complete();
    }
}


public interface IActivationFunction
{
    number Activate(number u);
    number Prime(number a);
}
public struct ReLU : IActivationFunction
{
    public number Activate(number u) => max(u, 0.0);
    public number Prime(number a) => sign(a);
}
public struct Sigmoid : IActivationFunction
{
    //public number Activate(number u) => 1.0 / (1.0 + exp(-u));
    public number Activate(number u)
    {
        var x = (1.0 + exp(-u));
        if (x == 0) return 0;
        return 1.0 / x;
    }
    public number Prime(number a) => a * (1.0 - a);
}
public struct Affine : IActivationFunction
{
    public number Activate(number u) => u;
    public number Prime(number a) => 1;
}
