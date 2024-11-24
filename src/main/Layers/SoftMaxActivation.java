import com.esotericsoftware.kryo.io.Input;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.loss.SoftmaxCrossEntropyLoss;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class SoftMaxActivation {
//    ActivationSoftmax
    public static INDArray softmax(INDArray input) {
        return Transforms.softmax(input,true);
    }

    public static INDArray d_softmax(INDArray softmaxOutput) {
        return Transforms.softmax(softmaxOutput,false);
    }
}
