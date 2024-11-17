import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear;
import org.nd4j.linalg.factory.Nd4j;

public class ConvolutionalNeuralNetwork {
    public static void main(String []args) {
        long []shape = {1000,28,28,3};
        INDArray A = Nd4j.rand(shape);
        long []Shape2 = {5,5,3};
        Conv2D C = new Conv2D(32,Shape2,0.0,0,1);
        System.out.println(C.forward(A));
    }
}