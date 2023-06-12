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

namespace nn.simd
{

    using number = Unity.Mathematics.float4;



    [System.Serializable]
    public struct NnActivations<T> : IDisposable where T:unmanaged
    {
        public NativeArray<T> currents;


        public int lengthOfNodes => this.currents.Length * unitlength;
        public int lengthOfUnits => this.currents.Length;

        public T this[int i]
        {
            get => this.currents[i];
            set => this.currents[i] = value;
        }


        unsafe static public int unitlength => sizeof(T) >> 2;
        static NativeArray<T> alloc(int length, Allocator allocator = Allocator.TempJob) =>
            new NativeArray<T>(length, allocator, NativeArrayOptions.UninitializedMemory);


        //public void SetNodeLength(int nodeLength) =>
        //    this.currents = alloc(nodeLength / unitlength, Allocator.);

        public NnActivations(int nodeLength) =>
            this.currents = alloc(nodeLength / unitlength, Allocator.Persistent);

        public NnActivations<T> CloneForTempJob() => new NnActivations<T>
        {
            currents = alloc(this.lengthOfUnits),
        };

        public void Dispose()
        {
            if (!this.currents.IsCreated) return;

            this.currents.Dispose();
        }
    }


    [System.Serializable]
    public struct NnWeights<T> : IDisposable where T : unmanaged
    {
        NativeArray<T> cn_x_p1;
        int width_n;


        public NativeArray<T> values => this.cn_x_p1;//

        public int widthOfNodes => this.width_n * unitlength;
        public int widthOfUnits => this.width_n;

        public int lengthOfUnits => this.cn_x_p1.Length;
        public int lengthOfNodes => this.cn_x_p1.Length * unitlength;

        public T this[int ic_n, int ip_1]
        {
            get => this.cn_x_p1[ip_1 * this.widthOfUnits + ic_n];
            set => this.cn_x_p1[ip_1 * this.widthOfUnits + ic_n] = value;
        }
        public T this[int i_n]
        {
            get => this.cn_x_p1[i_n];
            set => this.cn_x_p1[i_n] = value;
        }


        unsafe static public int unitlength => sizeof(T) >> 2;
        static NativeArray<T> alloc(int length, Allocator allocator = Allocator.TempJob) =>
            new NativeArray<T>(length, allocator, NativeArrayOptions.UninitializedMemory);


        public NnWeights(int prevLayerNodeLength, int currentLayerNodeLength)
        {
            var weightWidth = currentLayerNodeLength / unitlength;
            var weightHeight = prevLayerNodeLength + 1;
            var weightLength = weightWidth * weightHeight;

            this.cn_x_p1 = alloc(weightLength, Allocator.Persistent);
            this.width_n = weightWidth;
        }
        public NnWeights<T> CloneForTempJob() => new NnWeights<T>
        {
            cn_x_p1 = alloc(this.lengthOfUnits),
            width_n = this.width_n,
        };

        public void Dispose()
        {
            if (!this.cn_x_p1.IsCreated) return;

            this.cn_x_p1.Dispose();
        }
    }





    public interface IActivationFunction
    {
        number Activate(number u);
        number Prime(number a);
        void InitWeights(NnWeights<number> weights);
    }
    public struct ReLU : IActivationFunction
    {
        public number Activate(number u) => max(u, 0);
        public number Prime(number a) => sign(a);
        public void InitWeights(NnWeights<number> weights) => weights.InitHe();
    }
    public struct Sigmoid : IActivationFunction
    {
        public number Activate(number u) => 1 / (1 + exp(-u));
        public number Prime(number a) => a * (1 - a);
        public void InitWeights(NnWeights<number> weights) => weights.InitXivier();
    }
    public struct Affine4 : IActivationFunction
    {
        public number Activate(number u) => u;
        public number Prime(number a) => 1;
        public void InitWeights(NnWeights<number> weights) => weights.InitRandom();
    }
}