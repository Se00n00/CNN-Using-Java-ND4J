import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

public class MaxPool2D extends Layers{

    long[] WindowShape;
    double Lrate;
    int Strides;

    MaxPool2D(long[] Shape, double Lrate, int Strides){
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
        INDArray Output = Nd4j.create(OutputShape);

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

    @Override
    INDArray backward(INDArray Input) {

        return null;
    }
}