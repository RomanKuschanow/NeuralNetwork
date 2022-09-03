using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class Neuron
    {
        public List<decimal> Weights { get; set; }
        public decimal Bias { get; set; }
        public bool ChangeBias { get; set; }

        public Neuron(int weightsCount, decimal? bias = null, bool changeBias = true)
        {
            RandomSet(weightsCount);
            if (bias != null)
            {
                Bias = (decimal)bias;
            }
            ChangeBias = changeBias;
        }

        public Neuron(List<decimal> weights, decimal bias, bool changeBias = true)
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
            Weights = Enumerable.Range(0, weightsCount).Select(i => (decimal)r.NextDouble()).ToList();
            Bias = (decimal)r.NextDouble();
        }

        public decimal FeedForward(List<decimal> inputs)
        {
            if (inputs.Count != Weights.Count)
            {
                throw new Exception("The number of inputs must be the same as the number of weights");
            }
            return inputs.Select((input, i) => input * Weights[i]).Sum() + Bias;
        }

        public decimal SigmoidFeedForward(List<decimal> inputs)
        {
            return Sigmoid(FeedForward(inputs));
        }

        public void WeightsAndBiasUpdate(decimal ideal, List<decimal> inputs, decimal learnRate = 0.1m)
        {
            decimal derivS = DerivSigmoid(FeedForward(inputs));

            for (int i = 0; i < Weights.Count; i++)
            {
                Weights[i] -= learnRate * ideal * derivS  * inputs[i];
            }
            if (ChangeBias)
            {
                Bias -= learnRate * ideal * derivS;
            }

        }

        public static decimal DerivSigmoid(decimal x)
        {
            decimal fx = Sigmoid(x);
            return fx * (1 - fx);
        }

        public static decimal Sigmoid(decimal x)
        {
            decimal res = 1 / (decimal)(1 + Math.Exp((double)-x));
            return res;
        }
    }
}
