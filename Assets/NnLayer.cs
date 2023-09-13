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

    //using number = Unity.Mathematics.float4;
    //using nn.simd;


    [System.Serializable]
    public struct NnLayer<T> : IDisposable where T:unmanaged
    {
        public NnActivations<T> activations;
        public NnWeights<T> weights;

        public NnActivations<T> activations_delta;
        public NnWeights<T> weights_delta;

        public NnLayer(int nodeLength, int prevLayerNodeLength = 0)
        {
            this.activations = default;
            this.weights = default;
            this.activations_delta = default;
            this.weights_delta = default;

            this.activations = new NnActivations<T>(nodeLength);
            //this.activations.SetNodeLength(nodeLength);

            if (prevLayerNodeLength <= 0) return;

            this.weights = new NnWeights<T>(prevLayerNodeLength, nodeLength);
        }

        public void Dispose()
        {
            this.activations.Dispose();
            this.weights.Dispose();
            this.weights_delta.Dispose();
            this.activations_delta.Dispose();
        }
    }




    public static class Execute_NnLayerExtension
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


}