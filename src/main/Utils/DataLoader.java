import org.apache.commons.io.IOUtils;
import org.apache.hadoop.shaded.org.apache.kerby.util.IOUtil;
import org.deeplearning4j.core.loader.DataSetLoader;
import org.nd4j.common.loader.Source;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.InputStream;

public class DataLoader implements DataSetLoader {
    @Override
    public DataSet load(Source source) throws IOException {
        InputStream inputStream = source.getInputStream();
        byte[] data = IOUtils.toByteArray(inputStream);
//        INDArray features = Nd4j.create(data);
        return new DataSet();
    }
}
