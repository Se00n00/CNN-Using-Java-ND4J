import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class ConvolutionalNeuralNetwork {
    public static void main(String []args) {
        long []shape = {32,28,28,1};
        INDArray A = Nd4j.rand(shape);
        long []Shape2 = {3,3,1};
        long []Shape3 = {2,2};
        Conv2D C = new Conv2D(1,Shape2,0.0,0,1);
//        System.out.println(C.forward(A));
        INDArray Conv = C.forward(A);
        ReLU R = new ReLU();
        MaxPool2D M = new MaxPool2D(Shape3,0.0,1);
        INDArray Relud = R.relu(Conv);
        INDArray MaxPooled = M.forward(Relud);
        long[] MaxPooledShape = Arrays.stream(MaxPooled.shape()).toArray();
        int []ReshapeShape = {(int)MaxPooledShape[0],(int) MaxPooledShape[1] * (int) MaxPooledShape[2] * (int) MaxPooledShape[3]};
//        MaxPooled.reshape;
        System.out.println(Arrays.toString(ReshapeShape));
    }
}