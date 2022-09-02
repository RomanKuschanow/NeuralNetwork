using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class Neuron
    {
        public List<double> Weights { get; private set; }
        public double Bias { get; private set; }

        public Neuron(int weightsCount)
        {
            RandomSet(weightsCount);
        }

        public Neuron(List<double> weights, double bias)
        {
            Weights = weights;
            Bias = bias;
        }

        public void RandomSet()
        {
            RandomSet(Weights.Count);
        }

        private void RandomSet(int weightsCount)
        {
            Random r = new Random();
            Weights = Enumerable.Range(0, weightsCount).Select(i => r.NextDouble()).ToList();
            Bias = r.NextDouble();
        }

        public double FeedForward(List<double> inputs)
        {
            if (inputs.Count != Weights.Count)
            {
                throw new Exception("The number of inputs must be the same as the number of weights");
            }
            return inputs.Select((input, i) => input * Weights[i]).Sum() + Bias;
        }

        public double SigmoidFeedForward(List<double> inputs)
        {
            return Sigmoid(FeedForward(inputs));
        }

        public void WeightsAndBiasUpdate(List<double> dLdyPred, List<double> inputs, List<double> dyPreddh = null, double learnRate = 0.1)
        {
            double sum = FeedForward(inputs);
            double derivSum = DerivSigmoid(sum);

            for (int p = 0; p < dLdyPred.Count; p++)
            {
                for (int i = 0; i < Weights.Count; i++)
                {
                    double dhdw = inputs[i] * derivSum;
                    Weights[i] -= learnRate * dLdyPred[p] * (dyPreddh == null ? 1 : dyPreddh[p]) * dhdw;
                }
                Bias -= learnRate * dLdyPred[p] * (dyPreddh == null ? 1 : dyPreddh[p]) * derivSum;
            }
        }

        public static double DerivSigmoid(double x)
        {
            double fx = Sigmoid(x);
            return fx * (1 - fx);
        }

        public static double Sigmoid(double x)
        {
            double res = 1 / (1 + Math.Exp(-x));
            return res;
        }
    }
}
