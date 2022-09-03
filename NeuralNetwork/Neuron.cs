using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class Neuron
    {
        public List<double> Weights { get; set; }
        public double Bias { get; set; }
        public bool ChangeBias { get; set; }

        public Neuron(int weightsCount, double? bias = null, bool changeBias = true)
        {
            RandomSet(weightsCount);
            if (bias != null)
            {
                Bias = (double)bias;
            }
            ChangeBias = changeBias;
        }

        public Neuron(List<double> weights, double bias, bool changeBias = true)
        {
            Weights = weights;
            Bias = bias;
            ChangeBias = changeBias;
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

        public void WeightsAndBiasUpdate(double ideal, List<double> inputs, double learnRate = 0.1)
        {
            double derivS = DerivSigmoid(FeedForward(inputs));

            for (int i = 0; i < Weights.Count; i++)
            {
                Weights[i] -= learnRate * ideal * derivS  * inputs[i];
            }
            if (ChangeBias)
            {
                Bias -= learnRate * ideal * derivS;
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
