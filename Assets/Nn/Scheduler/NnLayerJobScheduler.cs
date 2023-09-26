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

    //using T = Unity.Mathematics.float4;
    //using nn.simd;

    public static partial class Nn<T, T1, Tc, Ta, Te, Td>
        where T : unmanaged
        where T1 : unmanaged
        where Tc : Calculation<T, Ta, Te, Td>
        where Ta : Calculation<T, Ta, Te, Td>.IForwardPropergationActivation, new()
        where Te : Calculation<T, Ta, Te, Td>.IBackPropergationError<Te>, new()
        where Td : Calculation<T, Ta, Te, Td>.IBackPropergationDelta<Td>, new()
    {



        public partial struct NnLayer
        {
            public JobHandle ExecuteUpdateWeightsJob(JobHandle dep)
            {
                return new NnLayerUpdateWeightsJob
                {
                    calc = Calc,
                    cxp_weithgs = this.weights,
                    cxp_weithgs_delta = this.weights_delta,
                }
                .Schedule(this.weights.lengthOfUnits, 64, dep);
            }
        }


        //static public void InitWeights<TAct>(this NnLayer layer)
        //    where TAct : struct, IActivationFunction
        //{
        //    var f = new TAct();
        //    foreach (var w in layer.weights.weights)
        //    {
        //        f.InitWeights(w);
        //    }
        //}



        public struct LayerPair
        {

            public NnLayer prev;
            public NnLayer curr;

            public LayerPair(NnLayer prev, NnLayer curr)
            {
                this.prev = prev;
                this.curr = curr;
            }

            public JobHandle ExecuteWithJob<Tact>(JobHandle dep)
                where Tact : struct, Calculation<T, Ta, Te, Td>.IActivationFunction
            {
                return new NnLayerForwardJob<Tact>
                {
                    prev_activations = this.prev.activations,
                    curr_activations = this.curr.activations,
                    cxp_weithgs = this.curr.weights,
                }
                .Schedule(this.curr.activations.lengthOfUnits, 1, dep);
            }

            public JobHandle ExecuteBackLastWithJob<Tact>(NativeArray<T> corrects, float learingRate, JobHandle dep)
                where Tact : struct, Calculation<T, Ta, Te, Td>.IActivationFunction
            {
                return new NnLayerBackLastJob<Tact>
                {
                    curr_activations = this.curr.activations,
                    curr_ds = this.curr.activations_delta,
                    curr_trains = corrects,
                    prev_activations = this.prev.activations,
                    dst_cxp_weithgs_delta = this.curr.weights_delta,
                    leaning_rate = learingRate,
                }
                .Schedule(this.curr.activations.lengthOfUnits, 1, dep);
            }
        }


        public struct Layer3Group
        {

            public NnLayer prev;
            public NnLayer curr;
            public NnLayer next;

            public Layer3Group(NnLayer prev, NnLayer curr, NnLayer next)
            {
                this.prev = prev;
                this.curr = curr;
                this.next = next;
            }

            public JobHandle ExecuteBackWithJob<Tact>(float learingRate, JobHandle dep)
                where Tact : struct, Calculation<T, Ta, Te, Td>.IActivationFunction
            {
                return new NnLayerBackJob<Tact>
                {
                    curr_activations = this.curr.activations,
                    curr_ds = this.curr.activations_delta,
                    prev_activations = this.prev.activations,
                    dst_cxp_weithgs_delta = this.curr.weights_delta,
                    nxc_weithgs = this.next.weights,
                    next_ds = this.next.activations_delta,
                    leaning_rate = learingRate,
                }
                .Schedule(this.curr.activations.lengthOfUnits, 1, dep);
            }
        }

    }


}