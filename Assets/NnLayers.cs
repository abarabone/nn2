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
            l.InitWeightToRandom();
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

    public void ExecuteBack(NativeArray<float> corrects, float learingRate)
    {
        {
            var prev = this.layers[this.layers.Length - 2];
            var curr = this.layers[this.layers.Length - 1];

            (prev, curr).ExecuteBackLast<Sigmoid>(corrects, learingRate);
        }

        for (var i = layers.Length - 1; i-- > 1;)
        {
            var prev = this.layers[i - 1];
            var curr = this.layers[i];
            var next = this.layers[i + 1];

            (prev, curr, next).ExecuteBack<ReLU>(learingRate);
        }
    }
    public void ExecuteBack2(NativeArray<float> corrects, float learingRate)
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
    float Activate(float u);
    float Prime(float a);
}
public struct ReLU : IActivationFunction
{
    public float Activate(float u) => max(u, 0.0f);
    public float Prime(float a) => sign(a);
}
public struct Sigmoid : IActivationFunction
{
    public float Activate(float u) => 1.0f / (1.0f + math.pow(math.E, -u));
    public float Prime(float a) => a * (1.0f - a);
}
