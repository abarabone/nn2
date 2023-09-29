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
    public static partial class Nn<T, T1, Tc, Ta, Te, Td>
        where T : unmanaged
        where T1 : unmanaged
        where Tc : ICalculation<T, Ta, Te, Td>, new()
        where Ta : ICalculation<T, Ta, Te, Td>.IForwardPropergationActivation, new()
        where Te : ICalculation<T, Ta, Te, Td>.IBackPropergationError<Te>, new()
        where Td : ICalculation<T, Ta, Te, Td>.IBackPropergationDelta<Td>, new()
    {


        [BurstCompile]
        public struct NnLayerForwardJob<TAct> : IJobParallelFor
            where TAct : struct, ICalculation<T, Ta, Te, Td>.IActivationFunction
        {
            [ReadOnly]
            public NnActivations<T> prev_activations;

            [WriteOnly]
            public NnActivations<T> curr_activations;

            [ReadOnly]
            public NnWeights<T> cxp_weithgs;


            [BurstCompile]
            public unsafe void Execute(int ic)
            {
                var sum = new Ta { value = default };

                var ip = 0;
                for (; ip < this.prev_activations.lengthOfNode; ip++)
                {
                    var a = this.prev_activations[ip];

                    sum.SumActivation(a, this.cxp_weithgs, ic, ip);
                }
                sum.SumBias(this.cxp_weithgs, ic, ip);

                this.curr_activations[ic] = new TAct().Activate(sum.value);

                //Debug.Log($"{ic} {sum.value} {new TAct().Activate(sum.value)}");
            }
        }


        // d  = -2(t - a) * a'
        // w -= d * a-
        [BurstCompile]
        public struct NnLayerBackLastJob<TAct> : IJobParallelFor
            where TAct : struct, ICalculation<T, Ta, Te, Td>.IActivationFunction
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


            [BurstCompile]
            public unsafe void Execute(int ic)
            {
                var t = this.curr_trains[ic];
                var o = this.curr_activations[ic];

                var err = new TAct().CalculateError(t, o);
                var prime = new TAct().Prime(o);

                var d = new Td().CalculateActivationDelta(err, prime, this.leaning_rate);
                this.curr_ds[ic] = d.raw;

                var ip = 0;
                for (; ip < this.prev_activations.lengthOfNode; ip++)
                {
                    var a_prev = this.prev_activations[ip];

                    d.SetWeightActivationDelta(this.dst_cxp_weithgs_delta, a_prev, ic, ip);
                }
                d.SetWeightBiasDelta(this.dst_cxp_weithgs_delta, ic, ip);
            }
        }

        // d  = sum(d+ * w+) * a'
        // w -= d * a-
        [BurstCompile]
        public struct NnLayerBackJob<TAct> : IJobParallelFor
            where TAct : struct, ICalculation<T, Ta, Te, Td>.IActivationFunction
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


            [BurstCompile]
            public unsafe void Execute(int ic)
            {
                var err = new Te { value = default };
                for (var inext = 0; inext < this.next_ds.lengthOfUnit; inext++)
                {
                    var nextActivationDelta = this.next_ds[inext];

                    err.SumActivationError(nextActivationDelta, this.nxc_weithgs, inext, ic);
                }
                var a = this.curr_activations[ic];
                var prime = new TAct().Prime(a);

                var d = new Td().CalculateActivationDelta(err.value, prime, this.leaning_rate);
                this.curr_ds[ic] = d.raw;

                var ip = 0;
                for (; ip < this.prev_activations.lengthOfNode; ip++)
                {
                    var a_prev = this.prev_activations[ip];

                    d.SetWeightActivationDelta(this.dst_cxp_weithgs_delta, a_prev, ic, ip);
                }
                d.SetWeightBiasDelta(this.dst_cxp_weithgs_delta, ic, ip);
            }
        }



        [BurstCompile]
        public struct NnLayerUpdateWeightsJob : IJobParallelFor
        {

            //[WriteOnly]
            public NnWeights<T> cxp_weithgs;

            [ReadOnly]//[DeallocateOnJobCompletion]
            public NnWeights<T> cxp_weithgs_delta;


            [BurstCompile]
            public void Execute(int i)
            {
                new Td().ApplyDeltaToWeight(this.cxp_weithgs, this.cxp_weithgs_delta, i);
            }
        }

    }
}