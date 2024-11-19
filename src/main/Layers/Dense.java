import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class Dense extends Layers{

    int Neurons;
    long TotalImages;
    double Lrate;
    INDArray Bias;
    INDArray Weights;
    long []WeightShape;
    INDArray Z;

    Dense(int Neurons, double Lrate){
        this.Neurons = Neurons;
        this.Lrate = Lrate;
    }

    @Override
    INDArray forward(INDArray Input){
        long[] InputShape = Arrays.stream(Input.shape()).toArray();

        // Initialize Parameters
//        TODO :: [1,Neurons] || [Neurons]
        this.TotalImages = InputShape[0];
        if(this.Bias == null)
            this.Bias = Nd4j.rand(1,this.Neurons);
        if(this.WeightShape == null)
            this.WeightShape = new long[]{InputShape[1], (long) this.Neurons};
        if(this.Weights == null)
            this.Weights = Nd4j.rand(this.WeightShape);

        System.out.println("[DENSE FORWARD PASS]");
        this.Z = Input.mmul(this.Weights).add(this.Bias);

        // Return Activation
        return Nd4j.getExecutioner().exec(new RectifiedLinear(this.Z));
    }

    INDArray backward(INDArray InputActivations, INDArray Weights, INDArray dL){
        System.out.println("[DENSE BACKWARD PASS]");
        ReLU rectifiedLU = new ReLU();
        INDArray dZ = dL.mmul(Weights.transpose()).mul(rectifiedLU.D_relu(this.Z));

        // Calculate Gradient for Weight and Bias
        INDArray dWeights = InputActivations.transpose().mmul(dZ).div(this.TotalImages);
        INDArray dBiases = dZ.sum(0).div(this.TotalImages);
        // Update Weights and Biases
        this.Weights = this.Weights.add(dWeights.mul(this.Lrate));
        this.Bias = this.Bias.add(dBiases.mul(this.Lrate));

        System.out.println("[DENSE BACKPASS COMPLETE]");
        return dZ;
    }
    public INDArray getWeights(){return this.Weights;}

}
