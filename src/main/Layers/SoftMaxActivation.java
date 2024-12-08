import com.esotericsoftware.kryo.io.Input;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.loss.SoftmaxCrossEntropyLoss;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class SoftMaxActivation {
    public static INDArray softmax(INDArray input) {
        return Nd4j.nn.softmax(input);
    }
    public static INDArray D_softmax(INDArray softmaxOutput) {
        return softmaxOutput.mul(softmaxOutput.rsub(1));
    }

}
