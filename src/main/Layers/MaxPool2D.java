import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

public class MaxPool2D extends Layers{

    long[] WindowShape;
    double Lrate;
    int Strides;
    INDArray Output;

    /**
     * Initialize Maximum Pooling Layer
     * @param WindowShape : Window-Shape > WindowShape X WindowShape
     * @param Strides : Jumps
     */
    MaxPool2D(long WindowShape, int Strides){
        this.WindowShape = new long[]{WindowShape,WindowShape};
        this.Strides = Strides;
    }

    @Override
    INDArray forward(INDArray Input){
        long []InputShape = Arrays.stream(Input.shape()).toArray();

        // Calculate the Output Shape
        long Window_i = (InputShape[1] - this.WindowShape[0])/this.Strides + 1;
        long Window_j = (InputShape[2] - this.WindowShape[1])/this.Strides + 1;
        long []OutputShape = {InputShape[0],Window_i,Window_j,InputShape[3]};

        // Create Output Index-N-Dim Array
        this.Output = Nd4j.create(OutputShape);

        // Compute the MaxPooling
        System.out.println("[MAXIMUM POOLING FORWARD PASS]"+Arrays.toString(Input.shape()));
        for(int b=0;b<InputShape[0];b++){
            for(int i=0;i<OutputShape[1];i++){
                for(int j=0;j<OutputShape[2];j++){
                    for(int k=0;k<OutputShape[3];k++){

                        INDArray InputPatch = Input.get(
                                NDArrayIndex.point(b),
                                NDArrayIndex.interval(i*this.Strides,i*this.Strides+this.WindowShape[0]),
                                NDArrayIndex.interval(j*this.Strides,j*this.Strides+this.WindowShape[1]),
                                NDArrayIndex.all()
                        );
                        Output.putScalar(new int[]{b,i,j,k}, (Double) InputPatch.maxNumber());
                    }
                }
            }
//            System.out.println("[BATCH---------------------------------------------]"+"["+(b+1)+"/"+InputShape[0]+"]");
        }

        // Return Maximum Pooled Output
        return Output;
    }

    INDArray backward(INDArray Input, INDArray dZ) {

        System.out.println("[MAXIMUM POOLING BACKWARD PASS]"+Arrays.toString(dZ.shape()));

        long[] inputShape = Input.shape();
        INDArray dP = Nd4j.zeros(inputShape);
        for (int b = 0; b < inputShape[0]; b++) {
            for (int h = 0; h < inputShape[1] - this.WindowShape[0]/this.Strides + 1; h++) {
                for (int w = 0; w < inputShape[2] - this.WindowShape[1]/this.Strides + 1; w++) {
                    for (int c = 0; c < inputShape[3]; c++) {

                        int hStart = h * this.Strides;
                        int wStart = w * this.Strides;

// Check bounds for window extraction
                        int hEnd = (int) Math.min(hStart + this.WindowShape[0], inputShape[1]);
                        int wEnd = (int) Math.min(wStart + this.WindowShape[1], inputShape[2]);

                        INDArray window = Input.get(
                                NDArrayIndex.point(b),
                                NDArrayIndex.interval(hStart, hEnd),
                                NDArrayIndex.interval(wStart, wEnd),
                                NDArrayIndex.all()
                        );


                        int[] maxIndex = Nd4j.argMax(window.reshape(1, -1), 1).toIntVector(); // Convert to 1D index
                        int maxH = maxIndex[0] / (int) this.WindowShape[0]; // Row index in window
                        int maxW = maxIndex[0] % (int) this.WindowShape[1]; // Column index in window

                        // Assign gradient to corresponding input index
                        double gradient = dZ.getDouble(b, h , w , c);
                        // Ensure the indices are within the bounds of the input array
                        int hIndex = (int) Math.min(h * this.Strides + maxH, inputShape[1] - 1);
                        int wIndex = (int) Math.min(w * this.Strides + maxW, inputShape[2] - 1);

                        dP.putScalar(new int[]{b, hIndex, wIndex, c}, gradient);

//                        dP.putScalar(new int[]{b, h*this.Strides + maxH, w*this.Strides + maxW, c}, gradient);
                    }
                }
            }
        }
        return dP;
    }

    public INDArray getOutput() {return Output;}
}