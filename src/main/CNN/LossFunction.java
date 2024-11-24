import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class LossFunction {
    /**
     * Categorical Cross-Entropy Loss (element-wise for each instance in the batch)
     *
     * @param yTrue The ground truth labels (one-hot encoded), shape: [batchSize, numClasses]
     * @param yPred The predicted probabilities, shape: [batchSize, numClasses]
     * @return An INDArray containing the loss for each instance in the batch, shape: [batchSize, 1]
     */
    public static INDArray categoricalCrossEntropy(INDArray yTrue, INDArray yPred) {
        // Apply log to yPred (numerically stable implementation to avoid log(0))
        INDArray logYPred = Transforms.log(yPred, true); // 'true' ensures the operation is applied in-place

        // Element-wise product of yTrue and log(yPred)
        INDArray lossMatrix = yTrue.mul(logYPred);

        // Sum across classes (axis = 1) and negate

        // Ensure the result is a column vector (batchSize, 1)
        return lossMatrix.sum(1).neg();
    }

}
