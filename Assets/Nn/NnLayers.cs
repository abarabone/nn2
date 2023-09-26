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
    public static partial class Nn<T, T1, Tc, Ta, Te, Td>
        where T : unmanaged
        where T1 : unmanaged
        where Tc : Calculation<T, Ta, Te, Td>
        where Ta : Calculation<T, Ta, Te, Td>.IForwardPropergationActivation, new()
        where Te : Calculation<T, Ta, Te, Td>.IBackPropergationError<Te>, new()
        where Td : Calculation<T, Ta, Te, Td>.IBackPropergationDelta<Td>, new()
    {


        [System.Serializable]
        public partial struct NnLayers : IDisposable
        {

            public NnLayer[] layers;


            public NnLayers(int[] nodeLengthList)
            {
                this.layers = nodeLengthList
                    .Prepend(0)
                    .SkipLast(1)
                    .Zip(nodeLengthList, (pre, cur) => new NnLayer(cur, pre))
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
}