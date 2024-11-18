import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.core.loader.impl.RecordReaderFileBatchLoader;
import org.deeplearning4j.spark.util.SparkDataUtils;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class DataSetPrepration {

    int BatchSize;
    int IMG_HEIGHTS;
    int IMG_WIDTHS = 32;
    int IMG_CHANNELS = 3;

    DataSetPrepration(int BatchSize,int IMG_HEIGHTS,int IMG_WIDTHS,int CHANNELS){
        this.BatchSize = BatchSize;
        this.IMG_HEIGHTS = IMG_HEIGHTS;
        this.IMG_WIDTHS = IMG_WIDTHS;
        this.IMG_CHANNELS = CHANNELS;
    }
    public void CreateDataset(String SourcePath,String OutputPath) throws IOException {

        File Source = new File(SourcePath);
        File Destination = new File(OutputPath);

        File[] Classes = Source.listFiles();
        String[] ClassList = new String[Classes.length];
        for(int i=0;i<Classes.length;i++){
            ClassList[i] = Classes[i].getName();
        }

        SparkDataUtils.createFileBatchesLocal(Source, NativeImageLoader.ALLOWED_FORMATS,true,Destination,this.BatchSize);
        PathLabelGenerator labelMaker =  new ParentPathLabelGenerator();
        ImageRecordReader rr = new ImageRecordReader(this.IMG_WIDTHS,this.IMG_HEIGHTS,this.IMG_CHANNELS,labelMaker);
        rr.setLabels(Arrays.asList(ClassList));
        int numClasses = ClassList.length;
        RecordReaderFileBatchLoader loader = new RecordReaderFileBatchLoader(rr,this.BatchSize,1,numClasses);
        loader.setPreProcessor(new ImagePreProcessingScaler());
    }

//    public static void main(String[] args) throws IOException {
//        String sourceDirectory = "/home/se00n00/Downloads/archive(3)/cifar10/cifar10/";
//        String destinationDirectory = "/home/se00n00/IdeaProjects/CNN-Using-Java/src/main/resources";
//
//        File Train = new File(sourceDirectory+"train");
//
//        File[] TrainClasses = Train.listFiles();
//
//        String[] ClassList = new String[TrainClasses.length];
//        for (int i=0;i<TrainClasses.length;i++) {
//            ClassList[i] = TrainClasses[i].getName();
//        }
//
//        File Source = new File(sourceDirectory);
//        File Destination = new File(destinationDirectory);
//
//        int BatchSize = 32;
//        SparkDataUtils.createFileBatchesLocal(Source, NativeImageLoader.ALLOWED_FORMATS,true,Destination,BatchSize);
//
//        int IMG_HEIGHTS = 32;
//        int IMG_WIDTHS = 32;
//        int IMG_CHANNELS = 3;
//
//        PathLabelGenerator labelMaker =  new ParentPathLabelGenerator();
//        ImageRecordReader rr = new ImageRecordReader(IMG_WIDTHS,IMG_HEIGHTS,IMG_CHANNELS,labelMaker);
//        rr.setLabels(Arrays.asList(ClassList));
//        int numClasses = ClassList.length;
//        RecordReaderFileBatchLoader loader = new RecordReaderFileBatchLoader(rr,BatchSize,1,numClasses);
//        loader.setPreProcessor(new ImagePreProcessingScaler());
//
//    }

}
