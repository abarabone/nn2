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
using static nn.ICalculation<T, Ta, Te, Td>;

namespace nn
{


    public interface ICalculation<T, Ta, Te, Td>
        //where U : Nn<U, T>.ICalculatable, new()
        where T : unmanaged
        where Ta : IForwardPropergationActivation, new()
        where Te : IBackPropergationError<Te>, new()
        where Td : IBackPropergationDelta<Td>, new()
    {

        //int UnitLength { get; }

        //V CreatActivation<V>() where V : IForwardPropergationActivation, new() => new V();
        //V CreatError<V>() where V : IBackPropergationError<V>, new() => new V();
        //V CreatDelta<V>() where V : IBackPropergationDelta<V>, new() => new V();

        //static public V CreateActivation<V>() where V : IForwardPropergationActivation, new() => new V();
        //static public V CreateError<V>() where V : IBackPropergationError<V>, new() => new V();
        //static public V CreateDelta<V>() where V : IBackPropergationDelta<V>, new() => new V();

        public Ta CreatActivation() => new Ta();
        public Te CreatError() => new Te();
        public Td CreatDelta() => new Td();


        public interface IForwardPropergationActivation
        {
            T value { get; set; }

            void SumActivation(T a, NnWeights<T> cxp_weithgs, int ic, int ip);
            void SumBias(NnWeights<T> cxp_weithgs, int ic, int ip);
        }

        public interface IBackPropergationError<V> where V : IBackPropergationError<V>
        {
            T value { get; set; }

            V CalculateError(T teach, T output);
            void SumActivationError(T nextActivationDelta, NnWeights<T> nxc_weithgs, int inext, int icurr);
        }
        public interface IBackPropergationDelta<V> where V : IBackPropergationDelta<V>
        {
            T rated { get; set; }
            T raw { get; set; }

            V CalculateActivationDelta(T err, T o_prime, float learningRate);

            void SetWeightActivationDelta(NnWeights<T> dst_cxp_weithgs_delta, T a, int ic, int ip);
            void SetWeightBiasDelta(NnWeights<T> dst_cxp_weithgs_delta, int ic, int ip);

            void ApplyDeltaToWeight(NnWeights<T> cxp_weithgs, NnWeights<T> cxp_weithgs_delta, int i);
        }
    }



    public static partial class Nn<U, T>
        where U : ICalculation<T>
        where T : unmanaged
    {
        //static public U CreateDelta(T value) => new U() { value = value };
        //static public U CreateSumValue(T value) => new U() { value = value };


        public interface IActivationFunction
        {
            T Activate(T u);
            T Prime(T a);
            void InitWeights(NnWeights<T> weights);

            T CalculateError(T t, T o);
        }

        public struct ReLU : IActivationFunction
        {
            public T Activate(T u) => max(u, 0);
            public T Prime(T a) => sign(a);

            public T CalculateError(T t, T o) => ;
            public void InitWeights(NnWeights<T> weights) => weights.InitHe();
        }
        public struct Sigmoid : IActivationFunction
        {
            public T Activate(T u) => 1 / (1 + exp(-u));
            public T Prime(T a) => a * (1 - a);

            public T CalculateError(T t, T o) => ;
            public void InitWeights(NnWeights<T> weights) => weights.InitXivier();
        }
        public struct Affine : IActivationFunction
        {
            public T Activate(T u) => u;
            public T Prime(T a) => 1;

            public T CalculateError(T t, T o) => ;
            public void InitWeights(NnWeights<T> weights) => weights.InitRandom();
        }



    }





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









    //public interface IActivationFunction
    //{
    //    number Activate(number u);
    //    number Prime(number a);
    //    void InitWeights(NnWeights<number> weights);
    //}

    //public struct ReLU : IActivationFunction
    //{
    //    public number Activate(number u) => max(u, 0);
    //    public number Prime(number a) => sign(a);
    //    public void InitWeights(NnWeights<number> weights) => weights.InitHe();
    //}
    //public struct Sigmoid : IActivationFunction
    //{
    //    public number Activate(number u) => 1 / (1 + exp(-u));
    //    public number Prime(number a) => a * (1 - a);
    //    public void InitWeights(NnWeights<number> weights) => weights.InitXivier();
    //}
    //public struct Affine : IActivationFunction
    //{
    //    public number Activate(number u) => u;
    //    public number Prime(number a) => 1;
    //    public void InitWeights(NnWeights<number> weights) => weights.InitRandom();
    //}

}