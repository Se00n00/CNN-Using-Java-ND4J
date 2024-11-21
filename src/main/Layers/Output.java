import org.apache.spark.sql.sources.In;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.dnn.Layer;

import java.util.Arrays;

public class Output extends Layers {

    int Neurons;
    long TotalImages;
    double Lrate;
    INDArray Bias;
    INDArray Weights;
    long []WeightShape;
    INDArray OutputResult;


    Output(int numClasses, double Lrate){
        this.Neurons = numClasses;
        this.Lrate = Lrate;
    }

    @Override
    INDArray forward(INDArray Input){
        long[] InputShape = Arrays.stream(Input.shape()).toArray();

        // Initialize Parameters
        this.TotalImages = InputShape[0];
//        TODO :: [1,Neurons] || [Neurons]
        if(this.Bias == null)
            this.Bias = Nd4j.rand(1,this.Neurons);
        if(this.WeightShape == null)
            this.WeightShape = new long[]{InputShape[1], (long) Neurons};
        if(this.Weights == null){
            double limit = Math.sqrt(6.0/Input.size(1) + this.Neurons);
            this.Weights = Nd4j.rand(this.WeightShape).muli(2*limit).subi(limit);
        }

        System.out.println("[OUTPUT DENSE FORWARD PASS]"+Arrays.toString(Input.shape()));

        // Return Activation
        ActivationSoftmax softmax = new ActivationSoftmax();
        this.OutputResult = softmax.getActivation(Input.mmul(this.Weights).add(this.Bias),true);
        return OutputResult;
    }

    INDArray backward(INDArray Activations, INDArray TrueLabels){
        System.out.println("[OUTPUT DENSE BACKWARD PASS]"+Arrays.toString(TrueLabels.shape()));
//        System.out.println(Arrays.toString(Activations.shape()));
//        TODO : TrueLabels - Activations ? or Activations - TrueLabels
        TrueLabels = TrueLabels.castTo(DataType.FLOAT);
        INDArray dZ = TrueLabels.sub(this.OutputResult);

        // Calculate Gradient for Weight and Bias
        INDArray dWeights = Activations.transpose().mmul(dZ).div(this.TotalImages);
        INDArray dBiases = dZ.sum(0).div(this.TotalImages);

        // Update Weights and Biases
        this.Weights = this.Weights.add(dWeights.mul(this.Lrate));
        this.Bias = this.Bias.add(dBiases.mul(this.Lrate));

        return dZ;
    }

    public INDArray getWeights() {return Weights;}
}
