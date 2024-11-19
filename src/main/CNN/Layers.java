import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Arrays;

public class Layers {

    INDArray forward(INDArray Input){
        return null;
    }

    INDArray backward(INDArray Input){
        return null;
    }


//    public static void main(String []args) {
//
//        long []shape = {32,28,28,1};
//        INDArray A = Nd4j.rand(shape);
//        long []Shape2 = {3,3,1};
//        long []Shape3 = {2,2};
//        Conv2D C = new Conv2D(1,Shape2,0.0,0,1);
////        System.out.println(C.forward(A));
//        INDArray Conv = C.forward(A);
//        ReLU R = new ReLU();
//        MaxPool2D M = new MaxPool2D(Shape3,0.0,1);
//        INDArray Relud = R.relu(Conv);
//        INDArray MaxPooled = M.forward(Relud);
//
//        long[] MaxPooledShape = Arrays.stream(MaxPooled.shape()).toArray();
//        long []ReshapeShape = {MaxPooledShape[0],MaxPooledShape[1] * MaxPooledShape[2] * MaxPooledShape[3]};
//
//        Dense D = new Dense(ReshapeShape,3, 0.0);
//        INDArray Maxed = MaxPooled.reshape(MaxPooledShape[0],MaxPooledShape[1] * MaxPooledShape[2] * MaxPooledShape[3]);
//        System.out.println(Arrays.toString(ReshapeShape));
//        System.out.println(Arrays.toString(D.WeightShape));
//        System.out.println(Arrays.toString(Arrays.stream(D.forward(Maxed).shape()).toArray()));
//    }

//    abstract INDArray backward(INDArray Input, INDArray Weights, INDArray dL);
}