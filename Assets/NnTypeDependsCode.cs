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

    using number = Unity.Mathematics.float4;
    using nn.simd;



    class Nna4
    {
        NnLayerForwardJob<ReLU> job1 = new();
        NnLayerForwardJob<Sigmoid> job2 = new();

        NnLayerBackLastJob<ReLU> job5 = new();
        NnLayerBackLastJob<Sigmoid> job6 = new();
        NnLayerBackJob<ReLU> job3 = new();
        NnLayerBackJob<Sigmoid> job4 = new();
    }



    public interface IActivationFunction
    {
        number Activate(number u);
        number Prime(number a);
        void InitWeights(NnWeights<number> weights);
    }

    //public struct ReLU : IActivationFunction
    //{
    //    public number Activate(number u) => max(u, 0);
    //    public number Prime(number a) => sign(a);
    //    public void InitWeights(NnWeights<number> weights) => weights.InitHe();
    //}
    //public struct Sigmoid : IActivationFunction
    //{
    //    public number Activate(number u) => 1 / (1 + exp(-u));
    //    public number Prime(number a) => a * (1 - a);
    //    public void InitWeights(NnWeights<number> weights) => weights.InitXivier();
    //}
    //public struct Affine : IActivationFunction
    //{
    //    public number Activate(number u) => u;
    //    public number Prime(number a) => 1;
    //    public void InitWeights(NnWeights<number> weights) => weights.InitRandom();
    //}




    static public class WeightExtension
    {


        static public unsafe void InitRandom(this NnWeights<number> ws)
        {
            var rnd = new NnRandom((uint)ws.values.GetUnsafePtr());

            for (var i = 0; i < ws.lengthOfUnits; i++)
            {
                ws[i] = rnd.Next();
            }
        }

        static public unsafe void InitXivier(this NnWeights<number> ws) =>
        ws.initX1X2((number)(1.0 / sqrt((double)ws.widthOfNodes)));

        static public unsafe void InitHe(this NnWeights<number> ws) =>
        ws.initX1X2((number)sqrt(2.0 / (double)ws.widthOfNodes));

        static unsafe void initX1X2(this NnWeights<number> ws, number std_deviation)
        {
            if (!ws.values.IsCreated) return;

            var rnd = new NnRandom((uint)ws.values.GetUnsafePtr());
            var pi2 = 2.0 * math.PI_DBL;

            (number x1, number x2) calc_x1x2() => (
                x1: (number)sqrt(((number)(-2.0) * (number)log(rnd.Next()))) * std_deviation,
                x2: (number)pi2 * rnd.Next());


            for (var io = 0; io < ws.lengthOfUnits >> 1; io++)
            {
                var (x1, x2) = calc_x1x2();

                var v0 =
                ws[io * 2 + 0] = x1 * cos(x2);
                //if (isnan(v0)) Debug.Log($"{io * 2 + 0} {v0}");
                //if (v0 == 0) Debug.Log($"{io * 2 + 0} {v0}");

                var v1 =
                ws[io * 2 + 1] = x1 * sin(x2);
                //if (isnan(v1)) Debug.Log($"{io * 2 + 1} {v1}");
                //if (v1 == 0) Debug.Log($"{io * 2 + 1} {v1}");
            }

            if ((ws.lengthOfUnits & 1) > 0)
            {
                var (x1, x2) = calc_x1x2();

                var v0 =
                ws[ws.lengthOfUnits - 1] = x1 * sin(x2);
                //if (isnan(v0)) Debug.Log($"{ws.length - 1} {v0}");
                //if (v0 == 0) Debug.Log($"{ws.length - 1} {v0}");
            }
        }
    }






    public static class NnLayerExtension
    {
        //static public void InitWeights<TAct>(this NnLayer layer)
        //    where TAct : struct, IActivationFunction
        //{
        //    var f = new TAct();
        //    foreach (var w in layer.weights.weights)
        //    {
        //        f.InitWeights(w);
        //    }
        //}


        static public JobHandle ExecuteWithJob<Tact>(this (NnLayer<number> prev, NnLayer<number> curr) layers, JobHandle dep)
            where Tact : struct, IActivationFunction
        {
            return new NnLayerForwardJob<Tact>
            {
                prev_activations = layers.prev.activations,
                curr_activations = layers.curr.activations,
                cxp_weithgs = layers.curr.weights,
            }
            .Schedule(layers.curr.activations.lengthOfUnits, 1, dep);
        }
        static public JobHandle ExecuteBackLastWithJob<Tact>(
            this (NnLayer<number> prev, NnLayer<number> curr) layers, NativeArray<number> corrects, float learingRate, JobHandle dep)
            where Tact : struct, IActivationFunction
        {
            return new NnLayerBackLastJob<Tact>
            {
                curr_activations = layers.curr.activations,
                curr_ds = layers.curr.activations_delta,
                curr_trains = corrects,
                prev_activations = layers.prev.activations,
                dst_cxp_weithgs_delta = layers.curr.weights_delta,
                leaning_rate = learingRate,
            }
            .Schedule(layers.curr.activations.lengthOfUnits, 1, dep);
        }
        static public JobHandle ExecuteBackWithJob<Tact>(
            this (NnLayer<number> prev, NnLayer<number> curr, NnLayer<number> next) layers, float learingRate, JobHandle dep)
            where Tact : struct, IActivationFunction
        {
            return new NnLayerBackJob<Tact>
            {
                curr_activations = layers.curr.activations,
                curr_ds = layers.curr.activations_delta,
                prev_activations = layers.prev.activations,
                dst_cxp_weithgs_delta = layers.curr.weights_delta,
                nxc_weithgs = layers.next.weights,
                next_ds = layers.next.activations_delta,
                leaning_rate = learingRate,
            }
            .Schedule(layers.curr.activations.lengthOfUnits, 1, dep);
        }

        static public JobHandle ExecuteUpdateWeightsJob(this NnLayer<number> layer, JobHandle dep)
        {
            return new NnLayerUpdateWeightsJob
            {
                cxp_weithgs = layer.weights,
                cxp_weithgs_delta = layer.weights_delta,
            }
            .Schedule(layer.weights.lengthOfUnits, 64, dep);
        }
    }




    public static class LayersExtension
    {

        public static JobHandle AddDeltaToWeightsWithDisposeTempJob(this NnLayers<number> layers, JobHandle dep)
        {
            for (var i = 0; i < layers.layers.Length; i++)
            {
                ref var l = ref layers.layers[i];

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
        void InitWeights(NnLayer<number>[] layers);

        JobHandle ExecuteForwardWithJob(NnLayer<number>[] layers, JobHandle deps = default);
        JobHandle ExecuteBackwordWithJob(NnLayer<number>[] layers, NativeArray<number> corrects, float learingRate, JobHandle deps = default);
    }


    public struct NnOtherAndLast<TOther, TLast> : INnExecutor
        where TOther : struct, IActivationFunction
        where TLast : struct, IActivationFunction
    {
        public void InitWeights(NnLayer<number>[] layers)
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

        public JobHandle ExecuteForwardWithJob(NnLayer<number>[] layers, JobHandle deps)
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
        public JobHandle ExecuteBackwordWithJob(
            NnLayer<number>[] layers, NativeArray<number> corrects, float learingRate, JobHandle deps)
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
