import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

public class Conv2D extends Layers{

    INDArray Bias;
    INDArray Weights;
    long[] ConvolutionShape;
    long[] WeightsShape;
    double Lrate;
    long Padding;
    long Strides;
    int Neurons;

    Conv2D(int Neurons,long[] ConvolutionShape,double Lrate,long Padding,long Strides){
        this.Neurons = Neurons;
        this.ConvolutionShape = ConvolutionShape.clone();
        this.Lrate = Lrate;
        this.Padding = Padding;
        this.Strides = Strides;
    }

    @Override
    INDArray forward(INDArray Input){
        long[] InputShape = Arrays.stream(Input.shape()).toArray();

        // Initiallize Parameter if Isn't Intitallized
        if(this.Bias == null)
            this.Bias = Nd4j.rand(this.Neurons);
        if(this.WeightsShape == null)
            this.WeightsShape = new long[]{this.ConvolutionShape[0],this.ConvolutionShape[1],InputShape[3],this.Neurons};

//        TODO: Xavier Intiallization
        if(this.Weights == null)
            this.Weights = Nd4j.rand(this.WeightsShape);


        // Check If Shape Is Correct
        if(InputShape.length != this.WeightsShape.length){
            System.out.println("Convolutional Forward Pass :: Input's Dimension "+InputShape.length+" Doesn't Matches to Kernel's Dimension "+ this.WeightsShape.length);
            System.out.println("");
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
        INDArray Output = Nd4j.zeros(OutputShape);

        // Compute the Convolution
        System.out.println("[CONVOLUTION FORWARD PASS]"+Arrays.toString(Input.shape()));
        for(int b=0;b<InputShape[0];b++){
            for(int i=0;i<OutputShape[1];i++){
                for(int j=0;j<OutputShape[2];j++){
                    for(int k=0;k<this.Neurons;k++){

                        INDArray InputPatch = Input.get(
                                NDArrayIndex.point(b),
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
        }

        // Apply Relu & Return Convolution Output
        ReluActivation R = new ReluActivation();
        return R.relu(Output);
    }

    INDArray backward(INDArray Input) {
        return null;
    }

    public INDArray getBias(){return this.Bias;}
    public INDArray getWeights(){return this.Weights;}
}