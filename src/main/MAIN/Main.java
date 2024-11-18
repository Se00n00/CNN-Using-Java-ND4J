import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.*;
import java.io.File;
import java.io.IOException;

public class Main {
    public static void IntitalDatasetCreation() throws IOException {

//        TODO : Change Content Between [ ] While Adding the New Image Path
//        TODO : [
        String TrainPath = "/home/se00n00/Downloads/archive(3)/cifar10/cifar10/train/";
        String TestPath = "/home/se00n00/Downloads/archive(3)/cifar10/cifar10/test/";

        String ResourceTrainPath = "/home/se00n00/IdeaProjects/CNN-Using-Java/src/main/resources/train/";
        String ResourceTestPath = "/home/se00n00/IdeaProjects/CNN-Using-Java/src/main/resources/test/";

        int BATCH_SIZE = 32;
        int IMG_HEIGHT = 32;
        int IMG_WIDTH = 32;
        int CHANNELS = 3;
        DataSetPrepration prepration = new DataSetPrepration(BATCH_SIZE,IMG_HEIGHT,IMG_WIDTH,CHANNELS);
        prepration.CreateDataset(TrainPath,ResourceTrainPath);
        prepration.CreateDataset(TestPath,ResourceTestPath);
//        TODO : ]

    }

    public static void main(String[] args) throws IOException {

//        TODO: Use Drive File for this -- Maybe

        File ResourcePath = new File("src/main/resources/train");

        // Checks If Resource Is Empty or Not
        if(ResourcePath.listFiles().length == 0){
            IntitalDatasetCreation(); // Creates Batches of Dataset
            System.out.println("[Data Preprocessing Complete]");
        }else{
            System.out.println("[Data Preprocessing Already Completed]");
        }

    }
}
