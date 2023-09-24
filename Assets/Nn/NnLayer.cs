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

    public static partial class Nn<T, T1, Ta, Te, Td>
        where T : unmanaged
        where T1 : unmanaged
        where Ta : Calculation<T, Ta, Te, Td>.IForwardPropergationActivation, new()
        where Te : Calculation<T, Ta, Te, Td>.IBackPropergationError<Te>, new()
        where Td : Calculation<T, Ta, Te, Td>.IBackPropergationDelta<Td>, new()
        //static public class NnLayerExtension
    {


        [System.Serializable]
        public partial struct NnLayer : IDisposable
        {
            public NnActivations<T> activations;
            public NnWeights<T> weights;

            public NnActivations<T> activations_delta;
            public NnWeights<T> weights_delta;

            public NnLayer(int nodeLength, int prevLayerNodeLength = 0)
            //public NnLayer(NnActivations<T> curr, NnActivations<T> prev = default)
            {
                this.activations = default;
                this.weights = default;
                this.activations_delta = default;
                this.weights_delta = default;

                this.activations = NnActivations<T>.Create<T1>(nodeLength);
                //this.activations.SetNodeLength(nodeLength);

                if (prevLayerNodeLength <= 0) return;

                this.weights = NnWeights<T>.Create<T1>(prevLayerNodeLength, nodeLength);
            }

            public void Dispose()
            {
                this.activations.Dispose();
                this.weights.Dispose();
                this.weights_delta.Dispose();
                this.activations_delta.Dispose();
            }
        }


    }
}