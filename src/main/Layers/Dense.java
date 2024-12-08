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
    private INDArray OutputActivations;

    Dense(int Neurons, String Activations){
        this.Neurons = Neurons;
        this.Activations = Activations;
    }

    /**
     * @param Input : Input Data in form of [Batches, Data]
     * @return Activations for Next Layer
     */
    @Override
    INDArray forward(INDArray Input){
        System.out.println("[DENSE FORWARD PASS]"+Arrays.toString(Input.shape()));
        long[] InputShape = Arrays.stream(Input.shape()).toArray();

        // Initialize Parameters
        this.TotalImages = InputShape[0];
        if(this.Bias == null)
            this.Bias = Nd4j.rand(1,this.Neurons);
        if(this.WeightShape == null)
            this.WeightShape = new long[]{InputShape[1], (long) this.Neurons};
        if(this.Weights == null){
            double limit = Math.sqrt(6.0/(Input.size(1) + this.Neurons));
            this.Weights = Nd4j.rand(this.WeightShape).muli(limit).subi(limit);      //Xavier Initialization
        }

        this.Z = Input.mmul(this.Weights).add(this.Bias);

        // Return Activation
        switch (this.Activations){
            case "RELU" :{
                this.OutputActivations = ReluActivation.relu(this.Z);
                return OutputActivations;
            }
            case "SOFTMAX" :{
                this.OutputActivations =  SoftMaxActivation.softmax(this.Z);
                return OutputActivations;
            }
            default:{
                System.out.println("[DENSE FORWARD PASS : INVALID ACTIVATION TYPO]");
                return null;
            }
        }
    }

    /**
     * @param dL : Loss for current Layer
     * @param InputActivations : Step-Back Layer Activations
     * @return dZ : Return Loss for Step-Back Layer
     */
    INDArray backward(INDArray dL, INDArray InputActivations){
        System.out.println("[DENSE BACKWARD PASS]"+Arrays.toString(dL.shape()));

        INDArray dZ = null;
        switch (this.Activations){
            case "RELU" :{
                dZ = dL.mul(ReluActivation.D_relu(this.Z));
                break;
            }
            case "SOFTMAX" :{
                dZ = dL;
                break;
            }
            default:{
                System.out.println("[DENSE BACKWARD PASS : INVALID ACTIVATION TYPE]");
                break;
            }
        }

        // Calculate Gradient for Weight and Bias
        INDArray dWeights = InputActivations.transpose().mmul(dZ).div(this.TotalImages);
        INDArray dBiases = dZ.sum(0).div(this.TotalImages);

        // Update Weights and Biases
        this.Weights = this.Weights.sub(dWeights.mul(this.Lrate));
        this.Bias = this.Bias.sub(dBiases.mul(this.Lrate));

        return dZ.mmul(getWeights().transpose());
    }
    public INDArray getWeights(){return this.Weights;}
    public INDArray getOutput(){return this.OutputActivations;}

}
