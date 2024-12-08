import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import org.imgscalr.Scalr;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Main {

    private static final Logger log = LoggerFactory.getLogger(Main.class);
    static String[] LABELS;
    static int BATCH_SIZE;
    static int IMG_HEIGHT = 32;
    static int IMG_WIDTH = 32;
    static int CHANNELS = 3;

    public static void main(String[] args) throws Exception {

        // Checks If Resource Is Empty or Not
        String ResourcePath = "src/main/resources";
        File Path = new File(ResourcePath);
        if(Path.listFiles().length == 0){
            Intital_Dataset_Creation(); // Creates Batches of Dataset
            System.out.println("[Data Preprocessing Complete]");
        }else{

            System.out.println("[Data Preprocessing Already Completed]");
        }

        String ResourceTrainPath = "src/main/resources/train/";
        String ResourceTestPath = "src/main/resources/test/";
        File TrainPath = new File(ResourceTrainPath);
        LABELS = new String[TrainPath.listFiles().length];
        int i = 0;
        System.out.print("CLASS LABELS :: ");
        for(File files:TrainPath.listFiles()){
            LABELS[i] = files.getName().toString();
            i++;
            System.out.print(files.getName()+" | ");
        }
        System.out.println();

        BATCH_SIZE = (int) 32/LABELS.length;

        // Create Train_X and Train_Y
        INDArray Train_X = null;
        INDArray Train_Y = null;
        for (String label : LABELS){
            File classFolder = new File(ResourceTrainPath,label);

            INDArray classImages = loadImagesFromDirectory(classFolder, false);
            INDArray classLabels = createLabelsArray(label, (int) classImages.size(0));

            if (Train_X == null) {
                Train_X = classImages;
            } else {
                Train_X = Nd4j.vstack(Train_X, classImages);
            }

            if (Train_Y == null) {
                Train_Y = classLabels;
            } else {
                Train_Y = Nd4j.vstack(Train_Y, classLabels);
            }
        }
        Train_X = Train_X.permute(0, 2, 3, 1);

        // Create Test_X and Test_Y
        INDArray Test_X = null;
        INDArray Test_Y = null;
        for (String label : LABELS){
            File classFolder = new File(ResourceTestPath,label);

            INDArray classImages = loadImagesFromDirectory(classFolder, false);
            INDArray classLabels = createLabelsArray(label, (int) classImages.size(0));

            if (Test_X == null) {
                Test_X = classImages;
            } else {
                Test_X = Nd4j.vstack(Test_X, classImages);
            }

            if (Test_Y == null) {
                Test_Y = classLabels;
            } else {
                Test_Y = Nd4j.vstack(Test_Y, classLabels);
            }
        }
        Test_X = Test_X.permute(0, 2, 3, 1);

        System.out.println("FINAL Train_X shape: " + Arrays.toString(Train_X.shape()));
        System.out.println("FINAL Train_Y shape: " + Arrays.toString(Train_Y.shape()));
        System.out.println("FINAL Train_X shape: " + Arrays.toString(Test_X.shape()));
        System.out.println("FINAL Train_Y shape: " + Arrays.toString(Test_Y.shape()));

//        TODO :: CHANGE THE ARCHITECTURE AS PER REQUIRED
        // METHOD 1 :: DEFINE THE MODEL
        NeuralNetwork NN1 = new NeuralNetwork();
        NN1.add(new Conv2D(10,7,0,4));
        NN1.add(new MaxPool2D(2,1));
        NN1.add(new Flatten());
        NN1.add(new Dense(128, "RELU"));
        NN1.add(new Dense(10,"SOFTMAX"));

        // METHOD 1 :: TRAIN THE MODEL
        NN1.fit(Train_X,Train_Y,0.001,10);

        // METHOD 1 :: EVALUATE THE MODEL
        System.out.println("Final Accuracy :: "+NN1.Accuracy(Test_X, Test_Y)+"100%");

        // METHOD 2 :: DEFINE THE MODEL
        NeuralNetwork NN2 = new NeuralNetwork(new ArrayList<>(Arrays.asList(
                new Conv2D(10,7,0,4),
                new MaxPool2D(2,1),
                new Dense(128, "RELU"),
                new Dense(10,"SOFTMAX")
        )));

        // METHOD 1 :: TRAIN THE MODEL
        NN1.fit(Train_X,Train_Y,0.001,10);

        // METHOD 1 :: EVALUATE THE MODEL
        System.out.println("Final Accuracy :: "+NN1.Accuracy(Test_X, Test_Y)+"100%");
    }

    public static void Intital_Dataset_Creation() throws IOException {

        String DataPath = "/home/se00n00/Downloads/archive(3)/cifar10/cifar10/train/";

        String ResourcePath = "src/main/resources/";

        DataSetPrepration prepration = new DataSetPrepration(IMG_HEIGHT,IMG_WIDTH);
        prepration.CreateDataset(DataPath,ResourcePath);
    }

    private static INDArray loadImagesFromDirectory(File folder, boolean augment) throws IOException {
        List<INDArray> imagesList = new ArrayList<>();
        NativeImageLoader loader = new NativeImageLoader(IMG_HEIGHT, IMG_WIDTH, CHANNELS);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);


        List<Integer> indices = IntStream.range(0, folder.listFiles().length).boxed().collect(Collectors.toList());
        Collections.shuffle(indices);
        List<File> selectedFiles = indices.stream().limit(BATCH_SIZE).map(i -> folder.listFiles()[i]).collect(Collectors.toList());

        for(File imgFile:selectedFiles){
            if (imgFile.isFile()) {
//                System.out.println("Loading image: " + imgFile.getName());
                BufferedImage originalImage = ImageIO.read(imgFile);
                if (originalImage != null) {
                    BufferedImage processedImage = augment ? augmentImage(originalImage) : originalImage;
                    INDArray image = loader.asMatrix(processedImage);
                    scaler.transform(image);
                    imagesList.add(image);
                }
            }
        }

        return Nd4j.vstack(imagesList);
    }

    private static INDArray createLabelsArray(String label, int numSamples) {
        int labelIndex = -1;
        for (int i = 0; i < LABELS.length; i++) {
            if (LABELS[i].equals(label)) {
                labelIndex = i;
                break;
            }
        }

        if (labelIndex == -1) {
            throw new IllegalArgumentException("Invalid label: " + label);
        }

        INDArray labels = Nd4j.zeros(numSamples, LABELS.length);
        for (int i = 0; i < numSamples; i++) {
            labels.putScalar(new int[]{i, labelIndex}, 1.0);
        }

        return labels;
    }

    private static BufferedImage augmentImage(BufferedImage image) {
        // Example augmentation: flip the image horizontally
        BufferedImage augmentedImage = Scalr.rotate(image, Scalr.Rotation.FLIP_HORZ);
        // Additional augmentation: rotate the image randomly
        double angle = new Random().nextDouble() * 360;
        augmentedImage = rotateImage(augmentedImage, angle);
        return augmentedImage;
    }

    private static BufferedImage rotateImage(BufferedImage image, double angle) {
        double radians = Math.toRadians(angle);
        double sin = Math.abs(Math.sin(radians));
        double cos = Math.abs(Math.cos(radians));
        int width = image.getWidth();
        int height = image.getHeight();
        int newWidth = (int) Math.floor(width * cos + height * sin);
        int newHeight = (int) Math.floor(height * cos + width * sin);

        BufferedImage rotatedImage = new BufferedImage(newWidth, newHeight, image.getType());
        Graphics2D g2d = rotatedImage.createGraphics();
        AffineTransform at = new AffineTransform();
        at.translate((newWidth - width) / 2, (newHeight - height) / 2);
        at.rotate(radians, width / 2, height / 2);
        g2d.setTransform(at);
        g2d.drawImage(image, 0, 0, null);
        g2d.dispose();
        return rotatedImage;
    }
}
