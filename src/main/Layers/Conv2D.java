import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

public class Conv2D {

    INDArray Bias;
    INDArray Weights;
    long[] WeightsShape;
    double Lrate;
    long Padding;
    long Strides;
    int Neurons;

    Conv2D(int Neurons,long[] shape,double Lrate,long Padding,long Strides){
        this.Bias = Nd4j.rand(Neurons);
        this.Neurons = Neurons;
//        TODO: Xavier Intiallization
//        this.Weights = Nd4j.rand(shape);
        this.WeightsShape = new long[]{shape[0],shape[1],shape[2],Neurons};
        this.Weights = Nd4j.rand(WeightsShape);
//        this.WeightsShape = Arrays.stream(this.Weights.shape()).toArray().clone();
        this.Lrate = Lrate;
        this.Padding = Padding;
        this.Strides = Strides;
    }
    INDArray forward(INDArray Input){

        long[] InputShape = Arrays.stream(Input.shape()).toArray();

        // Check If Channel Matches:
        if(InputShape.length != this.WeightsShape.length){
            System.out.println("Convolutional Forward Pass :: Input's Dimension "+InputShape.length+" Doesn't Matches to Kernel's Dimension "+ this.WeightsShape.length);
            return null;
        }
        if(InputShape[3] != this.WeightsShape[2]){
            System.out.println("Convolutional Forward Pass :: Input's Channel Doesn't Matches");
            return null;
        }
        if(InputShape[1] < Arrays.stream(this.Weights.shape()).toArray()[0] || InputShape[2] < Arrays.stream(this.Weights.shape()).toArray()[1]){
            System.out.println("Convolutional Forward Pass :: Input's Size Is too Less from Kernel Size");
            return null;
        }

        // Apply Padding
        Input = Nd4j.pad(Input,new int[][] {
                {0, 0},
                {(int) this.Padding, (int) this.Padding},
                {(int) this.Padding, (int) this.Padding},
                {0, 0}
        });

        // Calculate the Output Shape
        long kernal_i = (InputShape[1] - this.WeightsShape[0] + 2*this.Padding)/this.Strides + 1;
        long kernal_j = (InputShape[2] - this.WeightsShape[1] + 2*this.Padding)/this.Strides + 1;
        long[] OutputShape = {InputShape[0], kernal_i, kernal_j, this.Neurons};

        // Create Output Index-N-Dim Array
        INDArray Output = Nd4j.create(OutputShape);

        // Compute the Convolution
        System.out.println("[CONVOLUTION FORWARD PASS]");
        for(int b=0;b<InputShape[0];b++){
            for(int i=0;i<OutputShape[1];i++){
                for(int j=0;j<OutputShape[2];j++){
                    for(int k=0;k<this.Neurons;k++){

                        INDArray InputPatch = Input.get(NDArrayIndex.point(b),
                                NDArrayIndex.interval(i*this.Strides,i*this.Strides+WeightsShape[0]),
                                NDArrayIndex.interval(j*this.Strides,j*this.Strides+WeightsShape[1]),
                                NDArrayIndex.all()
                        );
                        INDArray KernalPath = this.Weights.get(NDArrayIndex.all(),
                                NDArrayIndex.all(),
                                NDArrayIndex.all(),
                                NDArrayIndex.point(k)
                        );

                        Output.putScalar(new int[]{b,i,j,k}, (Double) KernalPath.mul(InputPatch).sumNumber());
                    }
                }
            }
            System.out.println("[BATCH---------------------------------------------]"+"["+(b+1)+"/"+InputShape[0]+"]");
        }

        // Return Convolution Output
        return Output;
    }
    public INDArray getBias(){return this.Bias;}
    public INDArray getWeights(){
        return this.Weights;
    }
}