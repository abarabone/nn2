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
public struct NnLayer : IDisposable
{
    public NnActivations<number> activations;
    public NnWeights<number> weights;

    public NativeArray<number> deltas;//


    public NnLayer(int nodeLength, int prevLayerNodeLength = 0)
    {
        var nodeLengthWithBias = nodeLength + 1;

        this.activations = new NnActivations<number>
        {
            activations = new NativeArray<number>(
                nodeLengthWithBias,
                Allocator.Persistent,
                NativeArrayOptions.UninitializedMemory),
        };
        this.activations[this.activations.lengthOfNodes] = (number)1.0;   // for bias


        var weightWidth = prevLayerNodeLength + 1;
        var weightHeight = nodeLength;
        var weightLength = weightWidth * weightHeight;

        this.weights = prevLayerNodeLength > 0
            ? new NnWeights<number>
                {
                    weights = new NativeArray<number>(
                        weightLength,
                        Allocator.Persistent,
                        NativeArrayOptions.UninitializedMemory),

                    width = weightWidth,
                }
            : default
            ;


        this.deltas = new NativeArray<number>(
            nodeLength,
            Allocator.Persistent,
            NativeArrayOptions.UninitializedMemory);
    }

    public void Dispose()
    {
        this.activations.Dispose();
        this.weights.Dispose();
        if (this.deltas.IsCreated) this.deltas.Dispose();
    }
}


public static class NnLayerExtension
{

    static public void Execute<Tact>(this (NnLayer prev, NnLayer curr) layers)
        where Tact : struct, IActivationFunction
    {
        new NnLayerJob<Tact>
        {
            prev_activations = layers.prev.activations,
            curr_activations = layers.curr.activations,
            pxc_weithgs = layers.curr.weights,
        }
        .Run(layers.curr.activations.lengthOfNodes);
    }
    static public void ExecuteBackLast<Tact>(
        this (NnLayer prev, NnLayer curr) layers, NativeArray<number> corrects, float learingRate)
        where Tact : struct, IActivationFunction
    {
        new NnLayerBackLastJob<Tact>
        {
            curr_activations = layers.curr.activations,
            curr_ds = layers.curr.deltas,
            curr_trains = corrects,
            prev_activations = layers.prev.activations,
            pxc_weithgs = layers.curr.weights,
            leaning_rate = learingRate,
        }
        .Run(layers.curr.activations.lengthOfNodes);
    }
    static public void ExecuteBack<Tact>(
        this (NnLayer prev, NnLayer curr, NnLayer next) layers, float learingRate)
        where Tact : struct, IActivationFunction
    {
        new NnLayerBackJob<Tact>
        {
            curr_activations = layers.curr.activations,
            curr_ds = layers.curr.deltas,
            prev_activations = layers.prev.activations,
            pxc_weithgs = layers.curr.weights,
            cxn_weithgs = layers.next.weights,
            next_ds = layers.next.deltas,
            leaning_rate = learingRate,
        }
        .Run(layers.curr.activations.lengthOfNodes);
    }

    static public JobHandle Execute2<Tact>(this (NnLayer prev, NnLayer curr) layers, JobHandle dep)
        where Tact : struct, IActivationFunction
    {
        return new NnLayerJob<Tact>
        {
            prev_activations = layers.prev.activations,
            curr_activations = layers.curr.activations,
            pxc_weithgs = layers.curr.weights,
        }
        .Schedule(layers.curr.activations.lengthOfNodes, 1, dep);
    }
    static public JobHandle ExecuteBackLast2<Tact>(
        this (NnLayer prev, NnLayer curr) layers, NativeArray<number> corrects, float learingRate, JobHandle dep)
        where Tact : struct, IActivationFunction
    {
        return new NnLayerBackLastJob<Tact>
        {
            curr_activations = layers.curr.activations,
            curr_ds = layers.curr.deltas,
            curr_trains = corrects,
            prev_activations = layers.prev.activations,
            pxc_weithgs = layers.curr.weights,
            leaning_rate = learingRate,
        }
        .Schedule(layers.curr.activations.lengthOfNodes, 1, dep);
    }
    static public JobHandle ExecuteBack2<Tact>(
        this (NnLayer prev, NnLayer curr, NnLayer next) layers, float learingRate, JobHandle dep)
        where Tact : struct, IActivationFunction
    {
        return new NnLayerBackJob<Tact>
        {
            curr_activations = layers.curr.activations,
            curr_ds = layers.curr.deltas,
            prev_activations = layers.prev.activations,
            pxc_weithgs = layers.curr.weights,
            cxn_weithgs = layers.next.weights,
            next_ds = layers.next.deltas,
            leaning_rate = learingRate,
        }
        .Schedule(layers.curr.activations.lengthOfNodes, 1, dep);
    }
}

static public class WeightExtension
{
    static public unsafe void InitRandom(this NnWeights<number> ws)
    {
        var rnd = new Unity.Mathematics.Random((uint)ws.weights.GetUnsafePtr());

        for (var i = 0; i < ws.length; i++)
        {
            ws.weights[i] = rnd.NextFloat(0, 1);
        }
    }

    static public unsafe void InitXivier(this NnWeights<number> ws) =>
        ws.initX1X2(1.0 / sqrt(ws.widthOfNodes));

    static public unsafe void InitHe(this NnWeights<number> ws) =>
        ws.initX1X2(sqrt(2.0 / ws.widthOfNodes));

    static unsafe void initX1X2(this NnWeights<number> ws, number std_deviation)
    {
        if (!ws.weights.IsCreated) return;

        var rnd = new Unity.Mathematics.Random((uint)ws.weights.GetUnsafePtr());
        var pi2 = 2.0 * math.PI_DBL;

        (number x1, number x2) calc_x1x2() => (
            x1: sqrt(-2.0 * log(rnd.NextDouble())) * std_deviation,
            x2: pi2 * rnd.NextDouble());

        for (var io = 0; io < ws.length >> 1; io++)
        {
            var (x1, x2) = calc_x1x2();
            
            ws.weights[io * 2 + 0] = x1 * cos(x2);
            if (isnan(ws.weights[io * 2 + 0])) Debug.Log($"{io * 2 + 0} {ws.weights[io * 2 + 0]}");

            ws.weights[io * 2 + 1] = x1 * sin(x2);
            if (isnan(ws.weights[io * 2 + 1])) Debug.Log($"{io * 2 + 1} {ws.weights[io * 2 + 1]}");
        }

        if ((ws.length & 1) > 0)
        {
            var (x1, x2) = calc_x1x2();

            ws.weights[ws.length - 1] = x1 * sin(x2);
            if (isnan(ws.weights[ws.length - 1])) Debug.Log($"{ws.length - 1} {ws.weights[ws.length - 1]}");
        }
    }
}
