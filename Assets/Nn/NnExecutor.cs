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
using Unity.VisualScripting;

namespace nn
{

    static class e
    {
        public static IEnumerable<T> SkipLast<T>(this IEnumerable<T> e, int num) => Enumerable.SkipLast(e, num);
    }



    public static partial class Nn<T, T1, Tc, Ta, Te, Td>
        where T : unmanaged
        where T1 : unmanaged
        where Tc : ICalculation<T, Ta, Te, Td>, new()
        where Ta : ICalculation<T, Ta, Te, Td>.IForwardPropergationActivation, new()
        where Te : ICalculation<T, Ta, Te, Td>.IBackPropergationError<Te>, new()
        where Td : ICalculation<T, Ta, Te, Td>.IBackPropergationDelta<Td>, new()
    {


        public interface INnExecutor
        {
            void InitWeights(NnLayer[] layers);

            JobHandle ExecuteForwardWithJob(NnLayer[] layers, JobHandle deps = default);
            JobHandle ExecuteBackwordWithJob(NnLayer[] layers, NativeArray<T> corrects, float learingRate, JobHandle deps = default);
        }


        public struct NnOtherAndLast<TOther, TLast> : INnExecutor
            where TOther : struct, ICalculation<T, Ta, Te, Td>.IActivationFunction
            where TLast : struct, ICalculation<T, Ta, Te, Td>.IActivationFunction
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

                    deps = new LayerPair(prev, curr).ExecuteWithJob<TOther>(deps);
                }
                {
                    var prev = layers[layers.Length - 2];
                    var curr = layers[layers.Length - 1];

                    deps = new LayerPair(prev, curr).ExecuteWithJob<TLast>(deps);
                }

                return deps;
            }
            public JobHandle ExecuteBackwordWithJob(
                NnLayer[] layers, NativeArray<T> corrects, float learingRate, JobHandle deps)
            {
                {
                    ref var prev = ref layers[layers.Length - 2];
                    ref var curr = ref layers[layers.Length - 1];

                    deps = new LayerPair(prev, curr).ExecuteBackLastWithJob<TLast>(corrects, learingRate, deps);
                    //Debug.Log($"back last {this.layers.Length - 1}");
                }

                for (var i = layers.Length - 1; i-- > 1;)
                {
                    ref var prev = ref layers[i - 1];
                    ref var curr = ref layers[i];
                    ref var next = ref layers[i + 1];

                    deps = new Layer3Group(prev, curr, next).ExecuteBackWithJob<TOther>(learingRate, deps);
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

        ////    public JobHandle ExecuteBackwordWithJob(NnLayer[] layers, NativeArray<T> corrects, float learingRate, JobHandle deps) =>
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
        //    public void ExecuteBackword(NnLayer[] layers, NativeArray<T> corrects, float learingRate)
        //    {

        //    }

        //    public void ExecuteForwardWithJob(NnLayer[] layers)
        //    {

        //    }
        //    public void ExecuteBackwordWithJob(NnLayer[] layers, NativeArray<T> corrects, float learingRate)
        //    {

        //    }
        //}

    }
}
