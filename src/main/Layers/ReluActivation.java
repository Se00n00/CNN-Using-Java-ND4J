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
//        Nd4j.getExecutioner().exec(new Relu6(input));
//        INDArray mid = Transforms.max(input,0);
//        System.out.println("SHAPE"+ Arrays.toString(mid.shape()));
        return Transforms.relu(input,true);
    }
    public static INDArray D_relu(INDArray input) {
//        Nd4j.getExecutioner().exec(new Relu6Derivative());
//        INDArray derivative = input.dup();  // Duplicate the input to create the derivative
//        derivative = derivative.gt(0).castTo(Nd4j.defaultFloatingPointType()); // Set 1 where input > 0, else 0
        return Transforms.relu(input,false);
    }
}
