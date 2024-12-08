import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Evaluation {
    public static Double Accuracy(INDArray PredictedOutput, INDArray TrueOutput){
        TrueOutput = TrueOutput.castTo(DataType.FLOAT);
        if (!PredictedOutput.shapeInfoToString().equals(TrueOutput.shapeInfoToString())) {
            throw new IllegalArgumentException("ExpectedOutput and TrueOutput must have the same shape.");
        }

        // Index of Maximum Argument For Predicted-Output and True-Output
        INDArray predictedClasses = Nd4j.argMax(PredictedOutput,1);
        INDArray trueClasses = Nd4j.argMax(TrueOutput,1);

        INDArray matches = predictedClasses.eq(trueClasses);
        double correct = matches.castTo(DataType.FLOAT).sumNumber().doubleValue();
        double total = predictedClasses.length();

        return correct / total * 100.00;
    }
}
