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
    public struct NnLayers<T> : IDisposable where T:unmanaged
    {
        public NnLayer<T>[] layers;

        public NnLayers(int[] nodeLengthList)
        {
            this.layers = nodeLengthList
                .Prepend(0)
                .SkipLast(1)
                .Zip(nodeLengthList, (pre, cur) => new NnLayer<T>(cur, pre))
                .ToArray()
                ;
        }

        public void Dispose()
        {
            foreach (var l in this.layers)
            {
                l.Dispose();
            }
        }


        ////public void AllocActivationWorks()
        ////{
        ////    for (var i = 0; i < this.layers.Length; i++)
        ////    {
        ////        ref var l = ref this.layers[i];

        ////        l.activations = l.activations.CloneForTempJob();
        ////    }
        ////}
        public void AllocDeltaWorks()
        {
            for (var i = 0; i < this.layers.Length; i++)
            {
                ref var l = ref this.layers[i];

                l.activations_delta = l.activations.CloneForTempJob();
                l.weights_delta = l.weights.CloneForTempJob();
            }
        }

    }




}