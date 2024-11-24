import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

public class AvgPool2D extends Layers{

    long[] WindowShape;
    double Lrate;
    int Strides;
    INDArray Output;

    AvgPool2D(long []Shape,double Lrate,int Strides){
        this.WindowShape = Shape.clone();
        this.Lrate = Lrate;
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

        // Compute the AvgPooling
        System.out.println("[AVERAGE POOLING 2D FORWARD PASS]");
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
                        Output.putScalar(new int[]{b,i,j,k}, (Double) InputPatch.meanNumber());
                    }
                }
            }
            System.out.println("[BATCH---------------------------------------------]"+"["+(b+1)+"/"+InputShape[0]+"]");
        }

        // Return Average Pooled Output
        return Output;
    }

    INDArray backward(INDArray Input, INDArray dZ) {
        System.out.println("[AVERAGE POOLING BACKWARD PASS]"+Arrays.toString(Input.shape()));

//        TODO : Implement Average Pooling layer
        long[] inputShape = Input.shape();
        INDArray dP = Nd4j.zeros(inputShape);
        for (int b = 0; b < inputShape[0]; b++) {
            for (int h = 0; h < inputShape[1] - this.WindowShape[0]/this.Strides; h++) {
                for (int w = 0; w < inputShape[2] - this.WindowShape[1]/this.Strides; w++) {
                    for (int c = 0; c < inputShape[3]; c++) {

                        // Extract pooling window
                        INDArray window = Input.get(
                                NDArrayIndex.point(b),
                                NDArrayIndex.interval((long) h *this.Strides, (long) h *this.Strides+this.WindowShape[0]),
                                NDArrayIndex.interval((long) w *this.Strides, (long) w *this.Strides+this.WindowShape[1]),
                                NDArrayIndex.all()
                        );

                        int[] maxIndex = Nd4j.argMax(window.reshape(1, -1), 1).toIntVector(); // Convert to 1D index
                        int maxH = maxIndex[0] / (int) this.WindowShape[0]; // Row index in window
                        int maxW = maxIndex[0] % (int) this.WindowShape[1]; // Column index in window

                        // Assign gradient to corresponding input index
                        double gradient = dZ.getDouble(b, h , w , c);
                        dP.putScalar(new int[]{b, h*this.Strides + maxH, w*this.Strides + maxW, c}, gradient);
                    }
                }
            }
        }
        System.out.println("[CHECK MAX POOLING COMPLETED]"+Arrays.toString(dP.shape()));
        return dP;
    }

    public INDArray getOutput() {return Output;}
}