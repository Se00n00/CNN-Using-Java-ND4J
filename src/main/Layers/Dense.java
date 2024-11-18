import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear;
import org.nd4j.linalg.factory.Nd4j;

public class Dense extends Layers{

    int Neurons;
    long TotalImages;
    double Lrate;
    INDArray Bias;
    INDArray Weights;
    long []WeightShape;
    INDArray Z;

    Dense(long []InputShape, int Neurons, double Lrate){
        this.Neurons = Neurons;
        this.TotalImages = InputShape[0];
        this.Lrate = Lrate;
        this.Bias =  Nd4j.rand(1,Neurons);
        this.WeightShape = new long[]{InputShape[1], (long) Neurons};
        this.Weights = Nd4j.rand(this.WeightShape);
    }

    INDArray forward(INDArray Input){
        System.out.println("[DENSE FORWARD PASS]");
        this.Z = Input.mmul(this.Weights).add(this.Bias);

        // Return Activation
        return Nd4j.getExecutioner().exec(new RectifiedLinear(this.Z));
    }

    INDArray backward(INDArray Input, INDArray Weights, INDArray dL){
        ReLU rectifiedLU = new ReLU();
        INDArray dZ = dL.mmul(Weights.transpose()).mul(rectifiedLU.D_relu(this.Z));

        // Calculate Gradient for Weight and Bias
        INDArray dWeights = Input.transpose().mmul(dZ).div(this.TotalImages);
        INDArray dBiases = dZ.sum(0).div(this.TotalImages);

        // Update Weights and Biases
        this.Weights = this.Weights.add(dWeights.mul(this.Lrate));
        this.Bias = this.Bias.add(dBiases.mul(this.Lrate));

        return dZ;
    }

}
