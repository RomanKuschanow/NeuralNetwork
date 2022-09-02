using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class Network
    {
        private List<int> LaiersPreset { get; }
        private List<List<List<KeyValuePair<int, int>>>> NeuronConnetions { get; }
        public List<List<Neuron>> Neurons { get; }

        public Network(List<int> laiersPreset, List<KeyValuePair<KeyValuePair<int, int>, KeyValuePair<double, int>>> weight = null, List<KeyValuePair<KeyValuePair<int, int>, KeyValuePair<double, bool>>> bias = null, bool haveBias = true) : this(laiersPreset, laiersPreset.Skip(1)
            .Select((l, i) => Enumerable.Range(0, l)
                .Select(n => Enumerable.Range(0, laiersPreset[i])
                    .Select(c => new KeyValuePair<int, int>(i, c))
                    .ToList())
                .ToList())
            .ToList(), weight, bias, haveBias)
        { }

        public Network(List<int> laiersPreset, List<List<List<KeyValuePair<int, int>>>> neuronConnetions, List<KeyValuePair<KeyValuePair<int, int>, KeyValuePair<double, int>>> weight = null, List<KeyValuePair<KeyValuePair<int, int>, KeyValuePair<double, bool>>> bias = null, bool haveBias = true)
        {
            LaiersPreset = laiersPreset;
            NeuronConnetions = neuronConnetions;
            Neurons = laiersPreset.Skip(1).Select((l, i) => Enumerable.Range(0, l)
            .Select(n => new Neuron(NeuronConnetions[i][n].Count, haveBias ? null : 0, haveBias))
            .ToList()).ToList();

            if (bias != null)
            {
                EditBias(bias);
            }
            if (weight != null)
            {
                EditWeight(weight);
            }
        }

        public void EditBias(List<KeyValuePair<KeyValuePair<int, int>, KeyValuePair<double, bool>>> bias)
        {
            foreach (var item in bias)
            {
                try
                {
                    Neurons[item.Key.Key][item.Key.Value].Bias = item.Value.Key;
                    Neurons[item.Key.Key][item.Key.Value].ChangeBias = item.Value.Value;
                }
                catch
                {
                    continue;
                }
            }
        }

        public void EditWeight(List<KeyValuePair<KeyValuePair<int, int>, KeyValuePair<double, int>>> weight)
        {
            foreach (var item in weight)
            {
                try
                {
                    Neurons[item.Key.Key][item.Key.Value].Weights[item.Value.Value] = item.Value.Key;
                }
                catch
                {
                    continue;
                }
            }
        }

        #region Train
        public void Train(Dictionary<List<double>, List<double>> trainData, int epochs, double learnRate = 0.1)
        {
            for (int e = 0; e < epochs; e++)
            {
                foreach (var data in trainData)
                {
                    List<List<double>> pred = FeedForwardTrain(data.Key);
                    List<double> dLyPred = pred.Last().Select((p, i) => -2 * (data.Value[i] - Neuron.Sigmoid(p))).ToList();

                    Console.WriteLine(Math.Round(MSELoss(data.Value, pred.Last().Select(p => Neuron.Sigmoid(p)).ToList()), 4));

                    for (int l = 0; l < Neurons.Count; l++)
                    {
                        for (int n = 0; n < Neurons[l].Count; n++)
                        {
                            List<double> dyPreddh = new List<double>();

                            for (int _l = l + 1; _l < NeuronConnetions.Count; _l++)
                            {
                                for (int _n = 0; _n < NeuronConnetions[_l].Count; _n++)
                                {
                                    if (NeuronConnetions[_l][_n].Where(p => p.Key - 1 == l && p.Value == n).Count() > 0)
                                    {
                                        int weightIndex = NeuronConnetions[_l][_n].ToList().Select(c => c.Key == l + 1 && c.Value == n).ToList().IndexOf(true);
                                        dyPreddh.Add(Neurons[_l][_n].Weights[weightIndex] * Neuron.DerivSigmoid(pred[_l + 1][_n]));
                                    }
                                }
                            }

                            List<double> inputs = NeuronConnetions[l][n].Select(connetion => Neuron.Sigmoid(pred[connetion.Key][connetion.Value])).ToList();

                            Neurons[l][n].WeightsAndBiasUpdate(dLyPred, inputs, l + 1 < Neurons.Count ? dyPreddh : null, learnRate);
                        }
                    }
                }
            }
        }

        private List<List<double>> FeedForwardTrain(List<double> inputs)
        {
            List<List<double>> trainData = new List<List<double>>() { inputs };
            List<List<double>> data = new List<List<double>>() { inputs };

            for (int l = 0; l < Neurons.Count; l++)
            {
                data.Add(new List<double>());
                trainData.Add(new List<double>());

                for (int n = 0; n < Neurons[l].Count; n++)
                {
                    data[l + 1].Add(Neurons[l][n].SigmoidFeedForward(NeuronConnetions[l][n].Select(_n => data[_n.Key][_n.Value]).ToList()));
                    trainData[l + 1].Add(Neurons[l][n].FeedForward(NeuronConnetions[l][n].Select(_n => data[_n.Key][_n.Value]).ToList()));
                }
            }


            return trainData;
        }
        #endregion

        public List<double> FeedForward(List<double> inputs)
        {
            List<List<double>> data = new List<List<double>>() { inputs };

            for (int l = 0; l < Neurons.Count; l++)
            {
                data.Add(new List<double>());

                for (int n = 0; n < Neurons[l].Count; n++)
                {
                    data[l + 1].Add(Neurons[l][n].SigmoidFeedForward(NeuronConnetions[l][n].Select(_n => data[_n.Key][_n.Value]).ToList()));
                }
            }

            return data.Last();
        }

        public List<double> FeedBackward(List<double> inputs)
        {
            var neuronConnetions = new List<List<List<KeyValuePair<int, int>>>>();

            for (int l = 0; l < NeuronConnetions.Count; l++)
            {
                neuronConnetions.Insert(0, new List<List<KeyValuePair<int, int>>>());

                for (int n = 0; n < LaiersPreset[l]; n++)
                {
                    neuronConnetions[0].Add(new List<KeyValuePair<int, int>>());
                }
            }

            for (int l = 0; l < NeuronConnetions.Count; l++)
            {
                for (int n = 0; n < NeuronConnetions[NeuronConnetions.Count - 1 - l].Count; n++)
                {
                    for (int c = 0; c < NeuronConnetions[NeuronConnetions.Count - 1 - l][n].Count; c++)
                    {
                        neuronConnetions[l][c].Add(new KeyValuePair<int, int>(l, n));
                    }
                }
            }

            var network = new Network(LaiersPreset.Reverse<int>().ToList(), neuronConnetions, haveBias: false);

            for (int l = 0; l < Neurons.Count; l++)
            {
                for (int n = 0; n < Neurons[NeuronConnetions.Count - 1 - l].Count; n++)
                {
                    for (int w = 0; w < Neurons[NeuronConnetions.Count - 1 - l][n].Weights.Count; w++)
                    {

                    }
                }
            }

            return network.FeedForward(inputs);
        }

        public double MSELoss(List<double> yTrue, List<double> yPred)
        {
            if (yTrue.Count != yPred.Count)
            {
                throw new Exception("'yTrue' must have the same number of elements as 'yPred'");
            }

            return (1d / yTrue.Count) * Enumerable.Range(0, yTrue.Count).Select(i => Math.Pow(yTrue[i] - yPred[i], 2)).Sum();
        }
    }
}
