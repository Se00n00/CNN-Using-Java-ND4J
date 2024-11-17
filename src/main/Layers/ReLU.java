import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear;
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinearDerivative;
import org.nd4j.linalg.factory.Nd4j;

public class ReLU {

    public INDArray relu(INDArray Input){
        return  Nd4j.getExecutioner().exec(new RectifiedLinear(Input));
    }

    public INDArray D_relu(INDArray Input){
        return Input.gt(0).castTo(DataType.DOUBLE);
    }
}
