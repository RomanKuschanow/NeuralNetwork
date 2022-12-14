
using System;

namespace NeuralNetwork.Tests
{
    public class UnitTest1
    {
        [Fact]
        public void NeuralNetworkXORTest()
        {
            //Arrange
            var sut = new Network(new List<int>() { 2, 2, 1 });

            //Act

            var XOR = new Dictionary<List<decimal>, List<decimal>>()
            {
                { new List<decimal>() { 1, 1 },  new List<decimal>(){ 0 } },
                { new List<decimal>() { 1, 0 },  new List<decimal>(){ 1 } },
                { new List<decimal>() { 0, 1 },  new List<decimal>(){ 1 } },
                { new List<decimal>() { 0, 0 },  new List<decimal>(){ 0 } },
            };

            sut.Train(XOR, 5000, 10);

            var oneOne = sut.FeedForward(new List<decimal>() { 1, 1 })[0];
            var oneZero = sut.FeedForward(new List<decimal>() { 1, 0 })[0];
            var zeroOne = sut.FeedForward(new List<decimal>() { 0, 1 })[0];
            var zeroZero = sut.FeedForward(new List<decimal>() { 0, 0 })[0];

            //Assert
            Math.Round(oneOne).Should().Be(0);
            Math.Round(oneZero).Should().Be(1);
            Math.Round(zeroOne).Should().Be(1);
            Math.Round(zeroZero).Should().Be(0);

        }
    }
}
