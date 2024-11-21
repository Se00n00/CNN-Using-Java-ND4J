import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

public class SoftMaxActivation {
    public INDArray softmax(INDArray Input){
        INDArray expValues = Input.sub(Input.max());
        expValues = Transforms.exp(expValues);
        INDArray sumExp = expValues.sum(1).reshape(expValues.rows(), 1);
        return expValues.divColumnVector(sumExp);
    }
    public INDArray d_softmax(INDArray Input) {

        // Initialize Jacobian matrix
        int n = Input.columns();
        INDArray jacobian = Nd4j.create(Input.shape());

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    // Diagonal elements
                    jacobian.putScalar(i, j, Input.getDouble(i) * (1 - Input.getDouble(i)));
                } else {
                    // Off-diagonal elements
                    jacobian.putScalar(i, j, -Input.getDouble(i) * Input.getDouble(j));
                }
            }
        }
        return jacobian;
    }

}
