import org.apache.commons.math3.geometry.partitioning.Transform;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.loss.SoftmaxCrossEntropyLoss;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Test {
    public static void main(String[] args) {
        INDArray a = Nd4j.create(new double[][]{
                {2.0, -1.0, 0.1},
                {1.0, 2.0, 0.1},
                {-0.1, -0.2, 2.0}
        });
        SoftmaxCrossEntropyLoss s = new SoftmaxCrossEntropyLoss();

        System.out.println(Transforms.softmax(a,true));
        System.out.println(Transforms.softmax(a,false));
    }
}
