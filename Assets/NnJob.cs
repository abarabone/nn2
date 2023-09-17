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
using System.Numerics;

namespace nn
{

    //using number = Unity.Mathematics.float4;
    //using numbermt = Unity.Mathematics.float4x4;
    ////using Unity.VisualScripting;




    //class Nna4
    //{
    //    NnLayerForwardJob<ReLU> job1 = new();
    //    NnLayerForwardJob<Sigmoid> job2 = new();

    //    NnLayerBackLastJob<ReLU> job5 = new();
    //    NnLayerBackLastJob<Sigmoid> job6 = new();
    //    NnLayerBackJob<ReLU> job3 = new();
    //    NnLayerBackJob<Sigmoid> job4 = new();
    //}



    public static partial class Nn_<U, T>
        where U : Nn<U, T>.ICalculatable, new()
        where T : unmanaged
    {

        [BurstCompile]
        public struct NnLayerForwardJob<TAct> : IJobParallelFor
            where TAct : struct, IActivationFunction
        {
            [ReadOnly]
            public NnActivations<T> prev_activations;

            [WriteOnly]
            public NnActivations<T> curr_activations;

            [ReadOnly]
            public NnWeights<T> cxp_weithgs;


            public unsafe void Execute(int ic)
            {
                var sum = default(T);

                var ip = 0;
                for (; ip < this.prev_activations.lengthOfUnits; ip++)
                {
                    var a = this.prev_activations[ip];

                    new U().SumActivation(sum, a, this.cxp_weithgs, ic, ip);
                }
                new U().SumBias(sum, this.cxp_weithgs, ic, ip);

                this.curr_activations[ic] = new TAct().Activate(sum);
            }
        }


        // d  = -2(t - a) * a'
        // w -= d * a-
        [BurstCompile]
        public struct NnLayerBackLastJob<TAct> : IJobParallelFor
            where TAct : struct, IActivationFunction
        {
            // outputs
            [WriteOnly]
            public NnActivations<T> curr_ds;
            [WriteOnly]
            [NativeDisableParallelForRestriction]
            public NnWeights<T> dst_cxp_weithgs_delta;

            // for d calc
            [ReadOnly]
            public NativeArray<T> curr_trains;
            [ReadOnly]
            public NnActivations<T> curr_activations;

            // for w calc
            //[ReadOnly]//[DeallocateOnJobCompletion]
            //public NnWeights<number> cxp_weithgs;
            [ReadOnly]
            public NnActivations<T> prev_activations;

            public float leaning_rate;


            public unsafe void Execute(int ic)
            {
                var t = this.curr_trains[ic];
                var o = this.curr_activations[ic];
                var err = new U().CalculateError(t, o);
                var prime = new TAct().Prime(o);
                var d = new U().CalculateActivationDelta(err, prime, this.leaning_rate);

                this.curr_ds[ic] = d.raw;

                var ip = 0;
                for (; ip < this.prev_activations.lengthOfUnits; ip++)
                {
                    var a_prev = this.prev_activations[ip];

                    d.rated.SetWeightActivationDelta(this.dst_cxp_weithgs_delta, a_prev, ic, ip);
                }
                d.rated.SetWeightBiasDelta(this.dst_cxp_weithgs_delta, ic, ip);
            }
        }

        // d  = sum(d+ * w+) * a'
        // w -= d * a-
        [BurstCompile]
        public struct NnLayerBackJob<TAct> : IJobParallelFor
            where TAct : struct, IActivationFunction
        {
            // outputs
            [WriteOnly]
            public NnActivations<T> curr_ds;
            [WriteOnly]
            [NativeDisableParallelForRestriction]
            public NnWeights<T> dst_cxp_weithgs_delta;


            // for d calc
            [ReadOnly]
            public NnWeights<T> nxc_weithgs;
            [ReadOnly]
            public NnActivations<T> next_ds;
            [ReadOnly]
            public NnActivations<T> curr_activations;

            // for w calc
            //[ReadOnly]//[DeallocateOnJobCompletion]
            //public NnWeights<number> cxp_weithgs;
            [ReadOnly]
            public NnActivations<T> prev_activations;

            public float leaning_rate;


            public unsafe void Execute(int ic)
            {
                var err = new U();
                for (var inext = 0; inext < this.next_ds.lengthOfUnits; inext++)
                {
                    var nextActivationDelta = this.next_ds[inext];
                    err.SumActivationError(nextActivationDelta, this.nxc_weithgs, inext, ic);
                }
                var a = this.curr_activations[ic];
                var prime = new TAct().Prime(a);
                var d = new U().CalculateActivationDelta(err.value, prime, this.leaning_rate);

                this.curr_ds[ic] = d.raw;

                var ip = 0;
                for (; ip < this.prev_activations.lengthOfUnits; ip++)
                {
                    var a_prev = this.prev_activations[ip];

                    d.rated.SetWeightActivationDelta(this.dst_cxp_weithgs_delta, a_prev, ic, ip);
                }
                d.rated.SetWeightBiasDelta(this.dst_cxp_weithgs_delta, ic, ip);
            }
        }



        [BurstCompile]
        public struct NnLayerUpdateWeightsJob : IJobParallelFor
        {

            //[WriteOnly]
            public NnWeights<T> cxp_weithgs;

            [ReadOnly]//[DeallocateOnJobCompletion]
            public NnWeights<T> cxp_weithgs_delta;


            public void Execute(int i)
            {
                new U().ApplyDeltaToWeight(this.cxp_weithgs, this.cxp_weithgs_delta, i);
            }
        }
    }
}