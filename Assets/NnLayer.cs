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

namespace nn
{

    //using number = Unity.Mathematics.float4;
    //using nn.simd;


    [System.Serializable]
    public struct NnLayer<T> : IDisposable where T:unmanaged
    {
        public NnActivations<T> activations;
        public NnWeights<T> weights;

        public NnActivations<T> activations_delta;
        public NnWeights<T> weights_delta;

        public NnLayer(int nodeLength, int prevLayerNodeLength = 0)
        {
            this.activations = default;
            this.weights = default;
            this.activations_delta = default;
            this.weights_delta = default;

            this.activations = new NnActivations<T>(nodeLength);
            //this.activations.SetNodeLength(nodeLength);

            if (prevLayerNodeLength <= 0) return;

            this.weights = new NnWeights<T>(prevLayerNodeLength, nodeLength);
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