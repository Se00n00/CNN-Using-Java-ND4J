import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

public class Flatten extends Layers{

    long[] Conv2DenseShape;
    long[] Dense2ConvShape;

    @Override
    INDArray forward(INDArray Input) {
        long[] InputShape = Arrays.stream(Input.shape()).toArray();
        this.Dense2ConvShape = InputShape.clone();
        this.Conv2DenseShape = new long[]{InputShape[0], InputShape[1]*InputShape[2]*InputShape[3]};
        return Input.reshape(this.Conv2DenseShape);
    }

    @Override
    INDArray backward(INDArray Input) {
        System.out.println("[FLATTEN MIGHT NOT WORK :: FLATTEN BACKPROPAGATION]");
        return Input.reshape(this.Dense2ConvShape);
    }
}
