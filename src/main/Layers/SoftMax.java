import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class SoftMax {
    public INDArray softmax(INDArray Input){
        INDArray expValues = Input.sub(Input.max());
        expValues = Transforms.exp(expValues);
        INDArray sumExp = expValues.sum(1).reshape(expValues.rows(), 1);
        return expValues.divColumnVector(sumExp);
    }
}
