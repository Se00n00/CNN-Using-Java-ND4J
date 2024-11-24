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
        if(this.Weights == null){

            // He Initiallization
            double stddev = Math.sqrt(2.0 / (this.WeightsShape[0] * this.WeightsShape[1] * this.WeightsShape[2]));
            this.Weights = Nd4j.randn(this.WeightsShape).mul(stddev); // He Initialization

            // Xavier Initiallization
//            double limit = Math.sqrt(6.0/Input.size(1) + this.Neurons);
//            this.Weights = Nd4j.rand(this.WeightsShape).muli(2*limit).subi(limit);
        }


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
        return ReluActivation.relu(Output);
    }

    INDArray backward(INDArray Input, INDArray dZ) {
        System.out.println("[CONVOLUTIONAL BACKWARD PASS]" + Arrays.toString(dZ.shape()));

        long[] inputShape = Input.shape();
        long[] dZShape = dZ.shape();

        int batchSize = (int) inputShape[0];
        int inputHeight = (int) inputShape[1];
        int inputWidth = (int) inputShape[2];
        int inChannels = (int) inputShape[3];
        int dZHeight = (int) dZShape[1];
        int dZWidth = (int) dZShape[2];
        int outChannels = (int) dZShape[3];
        int filterHeight = (int) this.WeightsShape[0];
        int filterWidth = (int) this.WeightsShape[1];

        // Pad the input
        INDArray paddedInput = Nd4j.pad(Input, new int[][]{
                {0, 0},                                     // Batch dimension
                {(int) this.Padding, (int) this.Padding},   // Height padding
                {(int) this.Padding, (int) this.Padding},   // Width padding
                {0, 0}                                      // Channel dimension
        });

        // Initialize gradients
        INDArray dInput = Nd4j.zerosLike(paddedInput); // Gradient w.r.t input
        INDArray dFilter = Nd4j.zerosLike(this.Weights); // Gradient w.r.t filter
        INDArray dBias = dZ.sum(0, 1, 2); // Gradient w.r.t bias (sum over batch, height, and width)

        // Backpropagation
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < dZHeight; h++) {
                for (int w = 0; w < dZWidth; w++) {
                    int hStart = h * (int) this.Strides;
                    int hEnd = hStart + filterHeight;
                    int wStart = w * (int) this.Strides;
                    int wEnd = wStart + filterWidth;

                    for (int c = 0; c < outChannels; c++) {
                        // Slice input
                        INDArray inputSlice = paddedInput.get(
                                NDArrayIndex.point(b),
                                NDArrayIndex.interval(hStart, hEnd),
                                NDArrayIndex.interval(wStart, wEnd),
                                NDArrayIndex.all()
                        );

                        // Gradient w.r.t. filter
                        dFilter.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(c))
                                .addi(inputSlice.mul(dZ.getDouble(b, h, w, c)));

                        // Gradient w.r.t. input
                        dInput.get(
                                NDArrayIndex.point(b),
                                NDArrayIndex.interval(hStart, hEnd),
                                NDArrayIndex.interval(wStart, wEnd),
                                NDArrayIndex.all()
                        ).addi(this.Weights.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(c))
                                .mul(dZ.getDouble(b, h, w, c)));
                    }
                }
            }
        }

        // Update weights and biases
        this.Weights.subi(dFilter.mul(this.Lrate)); // Subtract gradient scaled by learning rate
        this.Bias.subi(dBias.mul(this.Lrate));     // Subtract gradient scaled by learning rate

        // Remove padding from dInput
        dInput = dInput.get(
                NDArrayIndex.all(),
                NDArrayIndex.interval((int) this.Padding, (int) this.Padding + inputHeight),
                NDArrayIndex.interval((int) this.Padding, (int) this.Padding + inputWidth),
                NDArrayIndex.all()
        );

        return dInput;
    }



    public INDArray getBias(){return this.Bias;}
    public INDArray getWeights(){return this.Weights;}
}