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

using number = System.Single;


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
        //var iw_start = this.prev_activations.lengthWithBias * curr_index;

        var sum = 0.0f;

        var ic = curr_index;
        for (var ip = 0; ip < this.prev_activations.lengthWithBias; ip++)
        {
            sum += this.prev_activations[ip] * this.pxc_weithgs[ip, ic];
        }

        this.curr_activations[curr_index] = new TAct().Activate(sum);
    }
}



//d2 = (t - a2)
//w2 -= d2 * a1

//d1 = d2 * a1'
//w1 -= d1 * a0



//d = (t - a)
//w -= d * a -
public struct NnLayerBackLastJob<TAct> : IJobParallelFor
    where TAct : struct, IActivationFunction
{
    // outputs
    [WriteOnly]
    public NativeArray<number> curr_ds;
    [NativeDisableParallelForRestriction]
    public NnWeights<number> pxc_weithgs;

    // for d calc
    [ReadOnly]
    public NativeArray<number> curr_trains;
    [ReadOnly]
    public NnActivations<number> curr_activations;

    // for w calc
    [ReadOnly]
    public NnActivations<number> prev_activations;

    public float leaning_rate;


    public void Execute(int curr_index)
    {
        var d = this.curr_activations[curr_index] - this.curr_trains[curr_index];
        d = d * new TAct().Prime(this.curr_activations[curr_index]);


        var ic = curr_index;
        //var iw_start = this.prev_activations.Length * curr_index;
        for (var ip = 0; ip < this.prev_activations.lengthWithBias; ip++)
        {
            this.pxc_weithgs[ip, ic] -= this.leaning_rate * d * this.prev_activations[ip];
        }

        this.curr_ds[curr_index] = d;
    }
}

//d  = sum(d+ * w+) * a'
//w -= d * a-
public struct NnLayerBackJob<TAct> : IJobParallelFor
    where TAct : struct, IActivationFunction
{
    // outputs
    [WriteOnly]
    public NativeArray<number> curr_ds;
    [NativeDisableParallelForRestriction]
    public NnWeights<number> pxc_weithgs;

    // for d calc
    [ReadOnly]
    public NnWeights<number> cxn_weithgs;
    [ReadOnly]
    public NativeArray<number> next_ds;
    [ReadOnly]
    public NnActivations<number> curr_activations;

    // for w calc
    [ReadOnly]
    public NnActivations<number> prev_activations;

    public float leaning_rate;


    public void Execute(int curr_index)
    {
        //var wcurr = this.curr_activations.Length;
        //var iw = curr_index;//this.curr_activations.Length; 

        var ic = curr_index;

        var sumd = 0.0f;
        for (var inext = 0; inext < this.next_ds.Length; inext++)
        {
            sumd += this.next_ds[inext] * this.cxn_weithgs[ic, inext];
        }
        var d = sumd * new TAct().Prime(this.curr_activations[curr_index]);


        //var iw_start = this.prev_activations.Length * curr_index;
        for (var ip = 0; ip < this.prev_activations.lengthWithBias; ip++)
        {
            this.pxc_weithgs[ip, ic] -= this.leaning_rate * d * this.prev_activations[ip];
        }

        this.curr_ds[curr_index] = d;
    }
}
