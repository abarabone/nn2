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
        public NnActivations activations;
        public NnWeights weights;

        public NativeArray<number> deltas;//
        public NnWeights weights_next_frame;

        public NnLayer(int nodeLength, int prevLayerNodeLength = 0)
        {
            var nodeLength4 = nodeLength >> 2;

            this.activations = new NnActivations
            {
                currents = new NativeArray<number>(
                    nodeLength4,
                    Allocator.Persistent,
                    NativeArrayOptions.UninitializedMemory),
            };


            var weightWidth4 = nodeLength >> 2;
            var weightHeight = prevLayerNodeLength + 1;
            var weightLength = weightWidth4 * weightHeight;

            this.weights = prevLayerNodeLength > 0
                ? new NnWeights
                {
                    c4xp = new NativeArray<number>(
                            weightLength,
                            Allocator.TempJob,
                            NativeArrayOptions.UninitializedMemory),

                    width4 = weightWidth4,
                }
                : default
                ;


            this.deltas = new NativeArray<number>(
                nodeLength4,
                Allocator.Persistent,
                NativeArrayOptions.UninitializedMemory);

            this.weights_next_frame = default;
        }

        public void Dispose()
        {
            this.activations.Dispose();
            this.weights.Dispose();
            this.weights_next_frame.Dispose();
            if (this.deltas.IsCreated) this.deltas.Dispose();
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
            return new NnLayerForward4Job<Tact>
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
            return new NnLayerBackLast4Job<Tact>
            {
                curr_activations = layers.curr.activations,
                curr_ds = layers.curr.deltas,
                curr_trains = corrects,
                prev_activations = layers.prev.activations,
                dst_cxp_weithgs = layers.curr.weights_next_frame,
                cxp_weithgs = layers.curr.weights,
                leaning_rate = learingRate,
            }
            .Schedule(layers.curr.activations.lengthOfUnits, 1, dep);
        }
        static public JobHandle ExecuteBackWithJob<Tact>(
            this (NnLayer prev, NnLayer curr, NnLayer next) layers, float learingRate, JobHandle dep)
            where Tact : struct, IActivationFunction
        {
            return new NnLayerBack4Job<Tact>
            {
                curr_activations = layers.curr.activations,
                curr_ds = layers.curr.deltas,
                prev_activations = layers.prev.activations,
                dst_cxp_weithgs = layers.curr.weights_next_frame,
                cxp_weithgs = layers.curr.weights,
                nxc_weithgs = layers.next.weights,
                next_ds = layers.next.deltas,
                leaning_rate = learingRate,
            }
            .Schedule(layers.curr.activations.lengthOfUnits, 1, dep);
        }
    }

    static public class WeightExtension
    {


        static public unsafe void InitRandom(this NnWeights ws)
        {
            var rnd = new Unity.Mathematics.Random((uint)ws.c4xp.GetUnsafePtr());

            for (var i = 0; i < ws.lengthOfUnits; i++)
            {
                ws.c4xp[i] = (number)rnd.NextDouble4(0, 1);
            }
        }

        static public unsafe void InitXivier(this NnWeights ws) =>
            ws.initX1X2((number)(1.0 / sqrt((double)ws.lengthOfNodes)));

        static public unsafe void InitHe(this NnWeights ws) =>
            ws.initX1X2((number)sqrt(2.0 / (double)ws.lengthOfNodes));

        static unsafe void initX1X2(this NnWeights ws, number std_deviation)
        {
            if (!ws.c4xp.IsCreated) return;

            var rnd = new Unity.Mathematics.Random((uint)ws.c4xp.GetUnsafePtr());
            var pi2 = 2.0 * math.PI_DBL;

            (number x1, number x2) calc_x1x2() => (
                x1: (number)sqrt(((number)(-2.0) * (number)log(rnd.NextDouble4()))) * std_deviation,
                x2: (number)(pi2 * rnd.NextDouble4()));

            for (var io = 0; io < ws.lengthOfUnits >> 1; io++)
            {
                var (x1, x2) = calc_x1x2();

                var v0 =
                ws.c4xp[io * 2 + 0] = x1 * cos(x2);
                //if (isnan(v0)) Debug.Log($"{io * 2 + 0} {v0}");
                //if (v0 == 0) Debug.Log($"{io * 2 + 0} {v0}");

                var v1 =
                ws.c4xp[io * 2 + 1] = x1 * sin(x2);
                //if (isnan(v1)) Debug.Log($"{io * 2 + 1} {v1}");
                //if (v1 == 0) Debug.Log($"{io * 2 + 1} {v1}");
            }

            if ((ws.lengthOfUnits & 1) > 0)
            {
                var (x1, x2) = calc_x1x2();

                var v0 =
                ws.c4xp[ws.lengthOfUnits - 1] = x1 * sin(x2);
                //if (isnan(v0)) Debug.Log($"{ws.length - 1} {v0}");
                //if (v0 == 0) Debug.Log($"{ws.length - 1} {v0}");
            }
        }
    }

}