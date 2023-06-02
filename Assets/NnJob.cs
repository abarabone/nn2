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

    using number = System.Single;
    //using Unity.VisualScripting;


    class Nna
    {
        NnLayerJob<ReLU> job1 = new();
        NnLayerJob<Sigmoid> job2 = new();

        NnLayerBackLastJob<ReLU> job5 = new();
        NnLayerBackLastJob<Sigmoid> job6 = new();
        NnLayerBackJob<ReLU> job3 = new();
        NnLayerBackJob<Sigmoid> job4 = new();
    }

    [BurstCompile]
    public struct NnLayerJob<TAct> : IJobParallelFor
        where TAct : struct, IActivationFunction
    {
        [ReadOnly]
        public NnActivations<number> prev_activations;

        [WriteOnly]
        public NnActivations<number> curr_activations;

        [ReadOnly]
        public NnWeights<number> pxc_weithgs;


        public void Execute(int curr_index)
        {
            var sum = default(number);

            var ip = 0;
            var ic = curr_index;
            for (; ip < this.prev_activations.lengthOfUnits; ip++)
            {
                sum += this.prev_activations[ip] * this.pxc_weithgs[ip, ic];
                //Debug.Log($"fwd p{ip} x c{ic} -> w:{this.pxc_weithgs[ip, ic]} * a-:{this.prev_activations[ip]}");
            }
            sum += this.pxc_weithgs[ip, ic];

            this.curr_activations[curr_index] = new TAct().Activate(sum);
        }
    }


    //d2 = (t - a2)
    //w2 -= d2 * a1

    //d1 = d2 * a1'
    //w1 -= d1 * a0


    // d  = -2(t - a) * a'
    // w -= d * a-
    [BurstCompile]
    public struct NnLayerBackLastJob<TAct> : IJobParallelFor
        where TAct : struct, IActivationFunction
    {
        // outputs
        [WriteOnly]
        public NativeArray<number> curr_ds;
        [WriteOnly]
        [NativeDisableParallelForRestriction]
        public NnWeights<number> dst_pxc_weithgs;

        // for d calc
        [ReadOnly]
        public NativeArray<number> curr_trains;
        [ReadOnly]
        public NnActivations<number> curr_activations;

        // for w calc
        [ReadOnly]//[DeallocateOnJobCompletion]
        public NnWeights<number> pxc_weithgs;
        [ReadOnly]
        public NnActivations<number> prev_activations;

        public float leaning_rate;


        public void Execute(int curr_index)
        {
            var t = this.curr_trains[curr_index];
            var o = this.curr_activations[curr_index];

            var d = -2 * (t - o);
            //Debug.Log($"back last d sum c{curr_index} x n{0} -> t:{t} a:{a} dl:{d}");
            d = d * new TAct().Prime(o);
            this.curr_ds[curr_index] = d;


            var drate = d * this.leaning_rate;

            var ip = 0;
            var ic = curr_index;
            for (; ip < this.prev_activations.lengthOfUnits; ip++)
            {
                this.dst_pxc_weithgs[ip, ic] = this.pxc_weithgs[ip, ic] - drate * this.prev_activations[ip];
                //Debug.Log($"back last sub weight p{ip} x c{ic} -> w:{this.pxc_weithgs[ip, ic]} a-:{this.prev_activations[ip]}");
            }
            this.dst_pxc_weithgs[ip, ic] = this.pxc_weithgs[ip, ic] - drate;
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
        public NativeArray<number> curr_ds;
        [WriteOnly]
        [NativeDisableParallelForRestriction]
        public NnWeights<number> dst_pxc_weithgs;


        // for d calc
        [ReadOnly]
        public NnWeights<number> cxn_weithgs;
        [ReadOnly]
        public NativeArray<number> next_ds;
        [ReadOnly]
        public NnActivations<number> curr_activations;

        // for w calc
        [ReadOnly]//[DeallocateOnJobCompletion]
        public NnWeights<number> pxc_weithgs;
        [ReadOnly]
        public NnActivations<number> prev_activations;

        public float leaning_rate;


        public void Execute(int curr_index)
        {
            var ic = curr_index;

            var sumd = default(number);
            for (var in_ = 0; in_ < this.next_ds.Length; in_++)
            {
                sumd += this.next_ds[in_] * this.cxn_weithgs[ic, in_];
                //Debug.Log($"back d sum c{ic} x n{in_} -> w:{this.cxn_weithgs[ic, in_]} * d+:{this.next_ds[in_]}");
            }
            var d = sumd * new TAct().Prime(this.curr_activations[curr_index]);
            this.curr_ds[curr_index] = d;


            var drate = d * this.leaning_rate;

            var ip = 0;
            for (; ip < this.prev_activations.lengthOfUnits; ip++)
            {
                this.dst_pxc_weithgs[ip, ic] = this.pxc_weithgs[ip, ic] - drate * this.prev_activations[ip];
                //Debug.Log($"back sub weight p{ip} x c{ic} -> w:{this.pxc_weithgs[ip, ic]} a-:{this.prev_activations[ip]}");
            }
            this.dst_pxc_weithgs[ip, ic] = this.pxc_weithgs[ip, ic] - drate;
        }
    }






}