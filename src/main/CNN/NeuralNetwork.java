import org.apache.spark.sql.sources.In;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private List<Layers> layers;
    public NeuralNetwork(ArrayList<Layers> LayerArray ) {
        this.layers = LayerArray;
    }
    public NeuralNetwork(){
        this.layers = new ArrayList<Layers>();
    }
    public void add(Layers layer){
        this.layers.add(layer);
    }

    public void fit(INDArray Train_X, INDArray Train_Y,double LearningRate, int Epochs){
        for(int e=0;e<Epochs;e++){

            // Forward Propagation
            INDArray Predictions = this.Forward(Train_X);

            // Compute Loss
            LossMCXENT lossFunction = new LossMCXENT();
            IActivation activation = new ActivationSoftmax();
//            System.out.println("LOSS :: " + lossFunction.computeScore(Train_Y.castTo(DataType.FLOAT), Predictions, activation,null,false));
//
            // Get Loss True_Output - Predicted_Output
            INDArray Loss = Train_Y.castTo(DataType.FLOAT).sub(Predictions);

            //Backward Propagation
            this.Backward(Loss, Train_X);

            // Evaluate and Print Accuracy
            System.out.println("EPOCH_____________________________________________________________["+(e+1)+"] [Accuracy :: "+this.Accuracy(Train_X,Train_Y)+"] "+"[LOSS :: "+lossFunction.computeScore(Train_Y.castTo(DataType.FLOAT), Predictions, activation,null,false)+"]");
        }
    }

    public INDArray Forward(INDArray Train_X){
        INDArray CurrentInput = Train_X;
        for(int i=0;i<layers.size();i++){
            CurrentInput = layers.get(i).forward(CurrentInput);
        }
        return CurrentInput;
    }
    public void Backward(INDArray Loss, INDArray Train_X){
        for(int i=layers.size()-1;i>=0;i--){
            if(layers.get(i) instanceof Dense && i!=0){
                Loss = ((Dense) layers.get(i)).backward(Loss,layers.get(i-1).getOutput());
            }else if(layers.get(i) instanceof Dense && i == 0){
                Loss = ((Dense) layers.get(i)).backward(Loss, Train_X);
            }else if(layers.get(i) instanceof Flatten){
                Loss = layers.get(i).backward(Loss);
            }else if(layers.get(i) instanceof MaxPool2D){
                Loss = ((MaxPool2D)layers.get(i)).backward(Loss, layers.get(i-1).getOutput());
            }else if(layers.get(i) instanceof Conv2D && i!=0){
                Loss = ((Conv2D)layers.get(i)).backward(Loss, layers.get(i-1).getOutput());
            }else if(layers.get(i) instanceof Conv2D && i==0){
                Loss = ((Conv2D)layers.get(i)).backward(Loss,Train_X);
            }
        }
    }

    public double Accuracy(INDArray TEST_X, INDArray TEST_Y){
        INDArray Predictions = this.Forward(TEST_X);
        return Evaluation.Accuracy(Predictions, TEST_Y);
    }

}
