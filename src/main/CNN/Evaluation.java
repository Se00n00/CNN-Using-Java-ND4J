import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Evaluation {
    public static Double Accuracy(INDArray ExpectedOutput, INDArray TrueOutput){
        TrueOutput = TrueOutput.castTo(DataType.FLOAT);
        if (!ExpectedOutput.shapeInfoToString().equals(TrueOutput.shapeInfoToString())) {
            throw new IllegalArgumentException("ExpectedOutput and TrueOutput must have the same shape.");
        }

        // Element-wise comparison
        INDArray matches = ExpectedOutput.eq(TrueOutput);

        // Sum up the correct matches
        double correct = matches.castTo(DataType.FLOAT).sumNumber().doubleValue();

        // Total number of elements
        double total = ExpectedOutput.length();

        return correct / total;
    }
}
