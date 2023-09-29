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
        where Tc : ICalculation<T, Ta, Te, Td>, new()
        where Ta : ICalculation<T, Ta, Te, Td>.IForwardPropergationActivation, new()
        where Te : ICalculation<T, Ta, Te, Td>.IBackPropergationError<Te>, new()
        where Td : ICalculation<T, Ta, Te, Td>.IBackPropergationDelta<Td>, new()
    {


        public partial struct NnLayers
        {

            public JobHandle AddDeltaToWeightsWithDisposeTempJob(JobHandle dep)
            {
                for (var i = 0; i < this.layers.Length; i++)
                {
                    ref var l = ref this.layers[i];

                    dep = l.ExecuteUpdateWeightsJob(dep);

                    //dep = l.activations.currents.Dispose(dep);
                    dep = l.activations_delta.currents.Dispose(dep);
                    dep = l.weights_delta.values.Dispose(dep);

                    l.activations_delta = default;
                    l.weights_delta = default;
                }
                return dep;
            }
            ////public JobHandle DisposeTempJobActivations(JobHandle dep)
            ////{
            ////    for (var i = 0; i < this.layers.Length; i++)
            ////    {
            ////        ref var l = ref this.layers[i];

            ////        //dep = l.activations.currents.Dispose(dep);
            ////        dep = l.activations_delta.currents.Dispose(dep);
            ////    }
            ////    return dep;
            ////}
            //public JobHandle DisposeTempJobAll(JobHandle dep)
            //{
            //    for (var i = 0; i < this.layers.Length; i++)
            //    {
            //        ref var l = ref this.layers[i];

            ////        dep = l.activations.currents.Dispose(dep);
            //        dep = l.activations_delta.currents.Dispose(dep);
            //        dep = l.weights_delta.values.Dispose(dep);
            //l.activations_delta = default;
            //        l.weights_delta = default;
            //    }
            //    return dep;
            //}
        }


    }
}