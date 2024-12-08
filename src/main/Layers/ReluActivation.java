import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear;
import org.nd4j.linalg.api.ops.impl.scalar.Relu6;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.Relu6Derivative;
import org.nd4j.linalg.factory.Nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

public class ReluActivation {
    public static INDArray relu(INDArray input) {
        return Transforms.relu(input,true);
    }
    public static INDArray D_relu(INDArray input) {
        return input.gt(0).castTo(input.dataType());
    }
}
