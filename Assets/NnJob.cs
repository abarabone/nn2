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

using number = System.Double;


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

        var ic = curr_index;
        for (var ip = 0; ip < this.prev_activations.lengthWithBias; ip++)
        {
            sum += this.prev_activations[ip] * this.pxc_weithgs[ip, ic];
            //Debug.Log($"fwd p{ip} x c{ic} -> w:{this.pxc_weithgs[ip, ic]} * a-:{this.prev_activations[ip]}");
        }

        this.curr_activations[curr_index] = new TAct().Activate(sum);
    }
}



//d2 = (t - a2)
//w2 -= d2 * a1

//d1 = d2 * a1'
//w1 -= d1 * a0


// d  = -2(t - a) * a'
// w -= d * a-
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
        var t = this.curr_trains[curr_index];
        var a = this.curr_activations[curr_index];

        var d = -2 * (t - a);
        //Debug.Log($"back last d sum c{curr_index} x n{0} -> t:{t} a:{a} dl:{d}");
        d = d * new TAct().Prime(a);


        var ic = curr_index;
        for (var ip = 0; ip < this.prev_activations.lengthWithBias; ip++)
        {
            this.pxc_weithgs[ip, ic] -= this.leaning_rate * d * this.prev_activations[ip];
            //Debug.Log($"back last sub weight p{ip} x c{ic} -> w:{this.pxc_weithgs[ip, ic]} a-:{this.prev_activations[ip]}");
        }

        this.curr_ds[curr_index] = d;
    }
}

// d  = sum(d+ * w+) * a'
// w -= d * a-
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
        var ic = curr_index;

        var sumd = default(number);
        for (var inext = 0; inext < this.next_ds.Length; inext++)
        {
            sumd += this.next_ds[inext] * this.cxn_weithgs[ic, inext];
            //Debug.Log($"back d sum c{ic} x n{inext} -> w:{this.cxn_weithgs[ic, inext]} * d+:{this.next_ds[inext]}");
        }
        var d = sumd * new TAct().Prime(this.curr_activations[curr_index]);


        for (var ip = 0; ip < this.prev_activations.lengthWithBias; ip++)
        {
            this.pxc_weithgs[ip, ic] -= this.leaning_rate * d * this.prev_activations[ip];
            //Debug.Log($"back sub weight p{ip} x c{ic} -> w:{this.pxc_weithgs[ip, ic]} a-:{this.prev_activations[ip]}");
        }

        this.curr_ds[curr_index] = d;
    }
}
