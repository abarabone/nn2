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

namespace nn.simd
{
    using number = Unity.Mathematics.float4;
    //using static UnityEditor.Experimental.GraphView.GraphView;


    [System.Serializable]
    public struct NnLayers : IDisposable
    {
        public NnLayer[] layers;

        public NnLayers(int[] nodeLengthList)
        {
            this.layers = nodeLengthList
                .Prepend(0)
                .SkipLast(1)
                .Zip(nodeLengthList, (pre, cur) => new NnLayer(cur, pre))
                .ToArray()
                ;
        }

        public void Dispose()
        {
            foreach (var l in this.layers)
            {
                l.Dispose();
            }
        }


        ////public void AllocActivationWorks()
        ////{
        ////    for (var i = 0; i < this.layers.Length; i++)
        ////    {
        ////        ref var l = ref this.layers[i];

        ////        l.activations = l.activations.CloneForTempJob();
        ////    }
        ////}
        public void AllocDeltaWorks()
        {
            for (var i = 0; i < this.layers.Length; i++)
            {
                ref var l = ref this.layers[i];

                l.activations_delta = l.activations.CloneForTempJob();
                l.weights_delta = l.weights.CloneForTempJob();
            }
        }

        public JobHandle AddDeltaToWeightsWithDisposeTempJob(JobHandle dep)
        {
            for (var i = 0; i < this.layers.Length; i++)
            {
                ref var l = ref this.layers[i];

                dep = l.ExecuteUpdateWeightsJob(dep);

                //dep = l.activations.currents.Dispose(dep);
                dep = l.activations_delta.currents.Dispose(dep);
                dep = l.weights_delta.values.Dispose(dep);

                l.activations_delta = default;
                l.weights_delta = default;
            }
            return dep;
        }
        ////public JobHandle DisposeTempJobActivations(JobHandle dep)
        ////{
        ////    for (var i = 0; i < this.layers.Length; i++)
        ////    {
        ////        ref var l = ref this.layers[i];

        ////        //dep = l.activations.currents.Dispose(dep);
        ////        dep = l.activations_delta.currents.Dispose(dep);
        ////    }
        ////    return dep;
        ////}
        //public JobHandle DisposeTempJobAll(JobHandle dep)
        //{
        //    for (var i = 0; i < this.layers.Length; i++)
        //    {
        //        ref var l = ref this.layers[i];

        ////        dep = l.activations.currents.Dispose(dep);
        //        dep = l.activations_delta.currents.Dispose(dep);
        //        dep = l.weights_delta.values.Dispose(dep);
        //l.activations_delta = default;
        //        l.weights_delta = default;
        //    }
        //    return dep;
        //}
    }





    public interface INnExecutor
    {
        void InitWeights(NnLayer[] layers);

        JobHandle ExecuteForwardWithJob(NnLayer[] layers, JobHandle deps = default);
        JobHandle ExecuteBackwordWithJob(NnLayer[] layers, NativeArray<number> corrects, float learingRate, JobHandle deps = default);
    }

    public struct NnOtherAndLast<TOther, TLast> : INnExecutor
        where TOther : struct, IActivationFunction
        where TLast : struct, IActivationFunction
    {
        public void InitWeights(NnLayer[] layers)
        {
            var actOther = new TOther();
            var actLast = new TLast();

            foreach (var l in layers.Skip(1).SkipLast(1))
            {
                actOther.InitWeights(l.weights);
            }
            {
                actLast.InitWeights(layers.Last().weights);
            }
        }

        public JobHandle ExecuteForwardWithJob(NnLayer[] layers, JobHandle deps)
        {
            for (var i = 1; i < layers.Length - 1; i++)
            {
                var prev = layers[i - 1];
                var curr = layers[i];

                deps = (prev, curr).ExecuteWithJob<TOther>(deps);
            }
            {
                var prev = layers[layers.Length - 2];
                var curr = layers[layers.Length - 1];

                deps = (prev, curr).ExecuteWithJob<TLast>(deps);
            }

            return deps;
        }
        public JobHandle ExecuteBackwordWithJob(NnLayer[] layers, NativeArray<number> corrects, float learingRate, JobHandle deps)
        {
            {
                ref var prev = ref layers[layers.Length - 2];
                ref var curr = ref layers[layers.Length - 1];

                deps = (prev, curr).ExecuteBackLastWithJob<TLast>(corrects, learingRate, deps);
                //Debug.Log($"back last {this.layers.Length - 1}");
            }

            for (var i = layers.Length - 1; i-- > 1;)
            {
                ref var prev = ref layers[i - 1];
                ref var curr = ref layers[i];
                ref var next = ref layers[i + 1];

                deps = (prev, curr, next).ExecuteBackWithJob<TOther>(learingRate, deps);
            }

            return deps;
        }
    }


    ////public struct NnUinform<TActivation> : INnExecutor
    ////    where TActivation : struct, IActivationFunction
    ////{

    ////    NnOtherAndLast<TActivation, TActivation> _base;


    ////    public void InitWeights(NnLayer[] layers) =>
    ////        this._base.InitWeights(layers);

    ////    public JobHandle ExecuteBackwordWithJob(NnLayer[] layers, NativeArray<number> corrects, float learingRate, JobHandle deps) =>
    ////        this._base.ExecuteBackwordWithJob(layers, corrects, learingRate, deps);

    ////    public JobHandle ExecuteForwardWithJob(NnLayer[] layers, JobHandle deps) =>
    ////        this._base.ExecuteForwardWithJob(layers, deps);
    ////}

    //public struct Nn : INnExecutor
    //{
    //    public void InitWeights(NnLayer[] layers)
    //    {

    //    }

    //    public void ExecuteForward(NnLayer[] layers)
    //    {

    //    }
    //    public void ExecuteBackword(NnLayer[] layers, NativeArray<number> corrects, float learingRate)
    //    {

    //    }

    //    public void ExecuteForwardWithJob(NnLayer[] layers)
    //    {

    //    }
    //    public void ExecuteBackwordWithJob(NnLayer[] layers, NativeArray<number> corrects, float learingRate)
    //    {

    //    }
    //}
}