import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import org.imgscalr.Scalr;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Main {

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
            IntitalDatasetCreation(); // Creates Batches of Dataset
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

//            System.out.println("Train_X shape: " + Arrays.toString(Train_X.shape()));
//            System.out.println("Train_Y shape: " + Arrays.toString(Train_Y.shape()));
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

//            System.out.println("Train_X shape: " + Arrays.toString(Test_X.shape()));
//            System.out.println("Train_Y shape: " + Arrays.toString(Test_Y.shape()));
        }
        Test_X = Test_X.permute(0, 2, 3, 1);

        System.out.println("FINAL Train_X shape: " + Arrays.toString(Train_X.shape()));
        System.out.println("FINAL Train_Y shape: " + Arrays.toString(Train_Y.shape()));
        System.out.println("FINAL Train_X shape: " + Arrays.toString(Test_X.shape()));
        System.out.println("FINAL Train_Y shape: " + Arrays.toString(Test_Y.shape()));


        // Define Model :: CURRENT : AlexNet
        long[] Convolution1 = new long[]{11,11};
        Conv2D C1 = new Conv2D(96,Convolution1,0.01,0,3);

        long[] Pooling1 = new long[]{3,3};
        MaxPool2D P1 = new MaxPool2D(Pooling1,0.01,1);

        long[] Convolution2 = new long[]{5,5};
        Conv2D C2 = new Conv2D(256, Convolution2,0.01,2,1);

        long[] Pooling2 = new long[]{3,3};
        MaxPool2D P2 = new MaxPool2D(Pooling2,0.01,2);

        long[] Convolution3 = new long[]{3,3};
        Conv2D C3 = new Conv2D(384, Convolution3,0.01,1,1);

        long[] Convolution4 = new long[]{3,3};
        Conv2D C4 = new Conv2D(384, Convolution4,0.01,1,1);

        long[] Convolution5 = new long[]{3,3};
        Conv2D C5 = new Conv2D(384, Convolution4,0.01,1,1);

        long[] Pooling3 = new long[]{3,3};
        MaxPool2D P3 = new MaxPool2D(Pooling2,0.01,2);

        Flatten F = new Flatten();

        Dense D1 = new Dense(20, 0.3);
        Output D2 = new Output(10, 0.03);

        for(int e = 0;e<5;e++){

            System.out.println("[EPOCH]--------------------------------------------------"+"["+(e+1)+"]");
            // Forward
            INDArray ConvolutionLayer = F.forward(P1.forward(C1.forward(Train_X)));
            INDArray Dense1Layer = D1.forward(ConvolutionLayer);
            INDArray Dense2Layer = D2.forward(Dense1Layer);

//            System.out.println(Arrays.toString(Dense2Layer.shape()));
            // Backward
            D1.backward(ConvolutionLayer,D2.getWeights(),D2.backward(Dense1Layer,Train_Y));

            // Evaluate Output
            System.out.println(Evaluation.Accuracy(Dense2Layer,Train_Y));
        }
//        System.out.println(Arrays.toString(D1.forward(ConvolutionalLayerOutput.reshape(Dense1)).shape()));


    }

    public static void IntitalDatasetCreation() throws IOException {

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
