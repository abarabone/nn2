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
    //using static Unity.Burst.Intrinsics.X86.Avx;

    [System.Serializable]
    public struct NnLayer : IDisposable
    {
        public NnActivations<number> activations;
        public NnWeights<number> weights;

        public NnActivations<number> activations_delta;
        public NnWeights<number> weights_delta;

        public NnLayer(int nodeLength, int prevLayerNodeLength = 0)
        {
            this.activations = default;
            this.weights = default;
            this.activations_delta = default;
            this.weights_delta = default;

            this.activations = new NnActivations<number>(nodeLength);
            //this.activations.SetNodeLength(nodeLength);

            if (prevLayerNodeLength <= 0) return;

            this.weights = new NnWeights<number>(prevLayerNodeLength, nodeLength);
        }

        public void Dispose()
        {
            this.activations.Dispose();
            this.weights.Dispose();
            this.weights_delta.Dispose();
            this.activations_delta.Dispose();
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


        static public JobHandle ExecuteWithJob<Tact>(this (NnLayer prev, NnLayer curr) layers, JobHandle dep)
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
            this (NnLayer prev, NnLayer curr) layers, NativeArray<number> corrects, float learingRate, JobHandle dep)
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
            this (NnLayer prev, NnLayer curr, NnLayer next) layers, float learingRate, JobHandle dep)
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

        static public JobHandle ExecuteUpdateWeightsJob(this NnLayer layer, JobHandle dep)
        {
            return new NnLayerUpdateWeightsJob
            {
                cxp_weithgs = layer.weights,
                cxp_weithgs_delta = layer.weights_delta,
            }
            .Schedule(layer.weights.lengthOfUnits, 64, dep);
        }
    }

    static public class WeightExtension
    {


        static public unsafe void InitRandom(this NnWeights<number> ws)
        {
            var rnd = new Unity.Mathematics.Random((uint)ws.values.GetUnsafePtr());

            for (var i = 0; i < ws.lengthOfUnits; i++)
            {
                ws[i] = (number)rnd.NextDouble4(0, 1);
            }
        }

        static public unsafe void InitXivier(this NnWeights<number> ws) =>
            ws.initX1X2((number)(1.0 / sqrt((double)ws.widthOfNodes)));

        static public unsafe void InitHe(this NnWeights<number> ws) =>
            ws.initX1X2((number)sqrt(2.0 / (double)ws.widthOfNodes));

        static unsafe void initX1X2(this NnWeights<number> ws, number std_deviation)
        {
            if (!ws.values.IsCreated) return;

            var rnd = new Unity.Mathematics.Random((uint)ws.values.GetUnsafePtr());
            var pi2 = 2.0 * math.PI_DBL;

            (number x1, number x2) calc_x1x2() => (
                x1: (number)sqrt(((number)(-2.0) * (number)log(rnd.NextDouble4()))) * std_deviation,
                x2: (number)(pi2 * rnd.NextDouble4()));

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

}