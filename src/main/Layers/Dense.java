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
    String Activations;

    Dense(int Neurons, double Lrate, String Activations){
        this.Neurons = Neurons;
        this.Lrate = Lrate;
        this.Activations = Activations;
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
        if(this.Weights == null){
            double limit = Math.sqrt(6.0/Input.size(1) + this.Neurons);
            this.Weights = Nd4j.rand(this.WeightShape).muli(2*limit).subi(limit);
        }

        System.out.println("[DENSE FORWARD PASS]"+Arrays.toString(Input.shape()));
        this.Z = Input.mmul(this.Weights).add(this.Bias);

        // Return Activation
        switch (this.Activations){
            case "RELU" :{
                return Nd4j.getExecutioner().exec(new RectifiedLinear(this.Z));
            }
            case "SOFTMAX" :{
                SoftMaxActivation S = new SoftMaxActivation();
                return S.softmax(this.Z);
            }
            default:{
                System.out.println("[DENSE FORWARD PASS : INVALID ACTIVATION TYPO]");
                return null;
            }
        }
    }

    INDArray backward(INDArray dL, INDArray InputActivations){
        System.out.println("[DENSE BACKWARD PASS]"+Arrays.toString(dL.shape()));

        System.out.println(Arrays.toString(InputActivations.shape())+Arrays.toString(Weights.shape()));
        ReluActivation rectifiedLU = new ReluActivation();

        INDArray dZ = null;
        switch (this.Activations){
            case "RELU" :{
                ReluActivation R = new ReluActivation();
                dZ = dL.mul(R.D_relu(this.Z));
                break;
            }
            case "SOFTMAX" :{
                SoftMaxActivation S = new SoftMaxActivation();
                dZ = dL.mul(S.d_softmax(this.Z));
                break;
            }
            default:{
                System.out.println("[DENSE BACKWARD PASS : INVALID ACTIVATION TYPE]");
                break;
            }
        }
//        INDArray dZ = dL.mul();
//        INDArray dZ = dL.mmul(Weights.transpose()).mul(rectifiedLU.D_relu(this.Z));

        // Calculate Gradient for Weight and Bias
        INDArray dWeights = InputActivations.transpose().mmul(dZ).div(this.TotalImages);
        INDArray dBiases = dZ.sum(0).div(this.TotalImages);

        // Update Weights and Biases
        this.Weights = this.Weights.add(dWeights.mul(this.Lrate));
        this.Bias = this.Bias.add(dBiases.mul(this.Lrate));

//        System.out.println("[DENSE BACKPASS COMPLETE]");
        return dZ.mmul(getWeights().transpose());
    }
    public INDArray getWeights(){return this.Weights;}

}
