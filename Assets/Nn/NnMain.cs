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


    public static partial class Nn<T, T1, Ta, Te, Td>
        where T : unmanaged
        where T1 : unmanaged
        where Ta : Calculation<T, Ta, Te, Td>.IForwardPropergationActivation, new()
        where Te : Calculation<T, Ta, Te, Td>.IBackPropergationError<Te>, new()
        where Td : Calculation<T, Ta, Te, Td>.IBackPropergationDelta<Td>, new()
    {

        public static Calculation<T, Ta, Te, Td> Calc;



    }



    // �ł��邾�����̉��������Ȃ��̂ŁA�v�Z�������� Nn ���番��
    public class Calculation<T, Ta, Te, Td>
        where T : unmanaged
        where Ta : Calculation<T, Ta, Te, Td>.IForwardPropergationActivation, new()
        where Te : Calculation<T, Ta, Te, Td>.IBackPropergationError<Te>, new()
        where Td : Calculation<T, Ta, Te, Td>.IBackPropergationDelta<Td>, new()
    {

        public Ta CreatActivation() => new Ta();
        public Te CreatError() => new Te();
        public Td CreatDelta() => new Td();

        public int nodesInUnit => new Ta().UnitLength;


        public interface IForwardPropergationActivation
        {
            T value { get; set; }
            int UnitLength { get; }

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


        public interface IActivationFunction
        {
            T Activate(T u);
            T Prime(T a);
            void InitWeights(NnWeights<T> weights);

            T CalculateError(T t, T o);
        }

        //public interface IWeightInitialize
        //{
        //    void InitRandom(NnWeights<T> ws);

        //    void InitXivier(NnWeights<T> ws);

        //    void InitHe(NnWeights<T> ws);
        //}
    }


}