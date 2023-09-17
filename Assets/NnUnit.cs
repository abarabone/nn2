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

    //interface Iex<T> where T:unmanaged
    //{

    //}
    //struct fex : Iex<float>
    //{

    //}

    //static class Ex
    //{
    //    static void a(this float a)
    //    {

    //    }
    //    static void b<T>(T a) where T:unmanaged
    //    {
    //        //a.a();
    //    }
    //    static void c()
    //    {
    //        var a = 1.0f;
    //        b(a);
    //    }
    //    static void d<T>(Iex<T> a) where T:unmanaged
    //    {

    //    }
    //    static void e()
    //    {
    //        var a = new fex();
    //        d(a);
    //    }
    //}






    //using Nn = Nn_<xfloat, float>;
    
    public struct xfloat : Nn<xfloat, float>.ICalculatable
    {
        public float value { get; set; }

        public int UnitLength => 1;


        public void SumActivation(float a, NnWeights<float> cxp_weithgs, int ic, int ip)
        {
            this.value += a * cxp_weithgs[ic, ip + 0];
        }
    }

    public struct xfloat4 : Nn<xfloat4, float4>.ICalculatable
    {
        public int UnitLength => 4;


        public struct ForwardActivation : Nn<xfloat4, float4>.ICalculatable.IForwardPropergationActivation
        {
            public float4 value { get; set; }

            public void SumActivation(float4 a, NnWeights<float4> cxp_weithgs, int ic, int ip)
            {
                var ip_ = ip * this.UnitLength;
                var ic_ = ic;

                this.value +=
                    a.xxxx * cxp_weithgs[ic_, ip_ + 0] +
                    a.yyyy * cxp_weithgs[ic_, ip_ + 1] +
                    a.zzzz * cxp_weithgs[ic_, ip_ + 2] +
                    a.wwww * cxp_weithgs[ic_, ip_ + 3];
            }

            public void SumBias(
                float4 a, NnWeights<float4> cxp_weithgs, int ic, int ip)
            {
                var ip_ = ip * this.UnitLength;

                this.value += cxp_weithgs[ic, ip_];
            }
        }


        public struct BackErorr : Nn<xfloat4, float4>.ICalculatable.IBackPropergationError
        {
            public float4 value { get; set; }

            // back last
            public float4 CalculateError(float4 teach, float4 output) =>
                -2.0f * (teach - output);


            // back other
            public void SumActivationError(
                float4 nextActivationDelta, NnWeights<float4> nxc_weithgs, int inext, int icurr)
            {
                var ic_ = icurr * this.UnitLength;

                var nd = nextActivationDelta;
                var dx = nd * nxc_weithgs[inext, ic_ + 0];
                var dy = nd * nxc_weithgs[inext, ic_ + 1];
                var dz = nd * nxc_weithgs[inext, ic_ + 2];
                var dw = nd * nxc_weithgs[inext, ic_ + 3];

                var md = new Matrix4x4(dx, dy, dz, dw);
                var tmd = math.transpose(md);

                this.value += tmd.c0 + tmd.c1 + tmd.c2 + tmd.c3;
            }
        }


        public struct BackDelta : Nn<xfloat4, float4>.ICalculatable.IBackPropergationDelta
        {
            public float4 value { get; set; }


            public (float4 raw, xfloat4 rated) CalculateActivationDelta(
            float4 err, float4 o_prime, float learningRate)
            {
                var d = err * o_prime;

                this.value = d * learningRate;

                return (d, this);
            }

            public void SetWeightActivationDelta(
                NnWeights<float4> dst_cxp_weithgs_delta, float4 a, int icurr, int iprev)
            {
                var ip_ = iprev * new Nn<xfloat4, float4>.ICalculatable().UnitLength;
                var ic = icurr;

                var delta_rated = this.value;
                dst_cxp_weithgs_delta[ic, ip_ + 0] = delta_rated * a.xxxx;
                dst_cxp_weithgs_delta[ic, ip_ + 1] = delta_rated * a.yyyy;
                dst_cxp_weithgs_delta[ic, ip_ + 2] = delta_rated * a.zzzz;
                dst_cxp_weithgs_delta[ic, ip_ + 3] = delta_rated * a.wwww;
            }
            public void SetWeightBiasDelta(
                NnWeights<float4> dst_cxp_weithgs_delta, int icurr, int iprev)
            {
                var ip_ = iprev * this.UnitLength;
                var ic = icurr;

                var delta_rated = this.value;
                dst_cxp_weithgs_delta[ic, ip_] = delta_rated;
            }
        }


        // 
        public void ApplyDeltaToWeight(NnWeights<T> cxp_weithgs, NnWeights<T> cxp_weithgs_delta, int i) =>
                cxp_weithgs[i] -= cxp_weithgs_delta[i];
    }




    public static partial class Nn<U, T>
        where U:Nn<U,T>.ICalculatable, new()
        where T:unmanaged
    {
        //static public U CreateDelta(T value) => new U() { value = value };
        //static public U CreateSumValue(T value) => new U() { value = value };

        public interface ICalculatable
        {
            int UnitLength { get; }

            public interface IForwardPropergationActivation
            {
                T value { get; set; }

                void SumActivation(T a, NnWeights<T> cxp_weithgs, int ic, int ip);
                void SumBias(NnWeights<T> cxp_weithgs, int ic, int ip);
            }
            public interface IBackPropergationError
            {
                T value { get; set; }

                T CalculateError(T teach, T output);
                void SumActivationError(T nextActivationDelta, NnWeights<T> nxc_weithgs, int inext, int icurr);
            }
            public interface IBackPropergationDelta
            {
                T value { get; set; }

                (T raw, U rated) CalculateActivationDelta(T err, T o_prime, float learningRate);

                void SetWeightActivationDelta(NnWeights<T> dst_cxp_weithgs_delta, T a, int ic, int ip);
                void SetWeightBiasDelta(NnWeights<T> dst_cxp_weithgs_delta, int ic, int ip);
            }


            void ApplyDeltaToWeight(NnWeights<T> cxp_weithgs, NnWeights<T> cxp_weithgs_delta, int i);
        }


        public interface IActivationFunction
        {
            T Activate(T u);
            T Prime(T a);
            //void InitWeights(NnWeights weights);

            T CalculateError(T t, T o);
        }

        //public interface IActivationFunction
        //{
        //    T Activate<T>(T u) where T : unmanaged;
        //    T Prime<T>(T a) where T : unmanaged;
        //    void InitWeights<T>(NnWeights<T> weights) where T : unmanaged;
        //}

        public struct ReLU : IActivationFunction
        {
            //public T Activate<T>(T u) where T : unmanaged => T.max(u, 0);
            //public T Prime<T>(T a) where T : unmanaged => sign(a);
            //public void InitWeights<T>(NnWeights<T> weights) where T : unmanaged => weights.InitHe();

            public T Activate(T u) => max(u, 0);
            public T Prime(T a) => sign(a);
            public float4 CalculateError() => ;
            public void InitWeights(NnWeights<U> weights) => weights.InitHe();
        }
        public struct Sigmoid : IActivationFunction
        {
            public T Activate<T>(T u) where T : unmanaged => 1 / (1 + exp(-u));
            public T Prime<T>(T a) where T : unmanaged => a * (1 - a);
            public void InitWeights<T>(NnWeights<T> weights) where T : unmanaged => weights.InitXivier();

            //public number Activate(number u) => 1 / (1 + exp(-u));
            //public number Prime(number a) => a * (1 - a);
            //public void InitWeights(NnWeights<number> weights) => weights.InitXivier();
        }
        public struct Affine : IActivationFunction
        {
            public T Activate<T>(T u) where T : unmanaged => u;
            public T Prime<T>(T a) where T : unmanaged => 1;
            public void InitWeights<T>(NnWeights<T> weights) where T : unmanaged => weights.InitRandom();

            //public number Activate(number u) => u;
            //public number Prime(number a) => 1;
            //public void InitWeights(NnWeights<number> weights) => weights.InitRandom();
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