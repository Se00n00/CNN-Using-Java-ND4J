import java.util.ArrayList;
import java.util.List;

public class CNN {
    private List<Object> layers;
    private double Lrate;

    public CNN(double learningRate,ArrayList<Object> LayerArray ) {
        this.Lrate = learningRate;
        this.layers = LayerArray;
    }
    public CNN(double learningRate){
        this.Lrate = learningRate;
        this.layers = new ArrayList<Object>();
    }
}
