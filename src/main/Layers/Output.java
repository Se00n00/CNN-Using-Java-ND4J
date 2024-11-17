import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;
import org.nd4j.linalg.factory.Nd4j;

public class Output {

    int Neurons;
    long TotalImages;
    double Lrate;
    INDArray Bias;
    INDArray Weights;
    long []WeightShape;
    INDArray Z;

    Output(long []InputShape, int Neurons, double Lrate){
        this.Neurons = Neurons;
        this.TotalImages = InputShape[0];
        this.Lrate = Lrate;
        this.Bias =  Nd4j.rand(1,Neurons);
        this.WeightShape = new long[]{InputShape[1], (long) Neurons};
        this.Weights = Nd4j.rand(this.WeightShape);
    }

    INDArray forward(INDArray Input){
        System.out.println("[OUTPUT DENSE FORWARD PASS]");
        this.Z = Input.mmul(this.Weights).add(this.Bias);

        // Return Activation
        ActivationSoftmax softmax = new ActivationSoftmax();
        return softmax.getActivation(this.Z.dup(),true);
    }

    INDArray backward(INDArray Input, INDArray Activations, INDArray ExpectedOutput){
        INDArray dZ = ExpectedOutput.sub(Activations);

        // Calculate Gradient for Weight and Bias
        INDArray dWeights = Input.transpose().mmul(dZ).div(this.TotalImages);
        INDArray dBiases = dZ.sum(0).div(this.TotalImages);

        // Update Weights and Biases
        this.Weights = this.Weights.add(dWeights.mul(this.Lrate));
        this.Bias = this.Bias.add(dBiases.mul(this.Lrate));

        return dZ;
    }

}
