package com.mytest_ym;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Hello world!
 *
 */
public class App 
{
    private static Logger log = LoggerFactory.getLogger(App.class);
    public static void main( String[] args ) throws Exception
    {
        //System.out.println( "Hello World!" );
        //INDArray nd = Nd4j.create( new float[]{1, 2, 3, 4}, new int[]{2, 2} );
        //System.out.println( nd );

        // construct dataset

        log.info("Load data...");
        String dataPath = System.getProperty("user.home");
        dataPath += "/Public/PATTERN_RECOGNITION/dataset/";
        log.info(dataPath);

        List<String> imageLabels = new ArrayList<String>();

        for(File f : new File(dataPath).listFiles()){
            imageLabels.add(f.getName());
        }

        log.info("Image Labels:");
        for(String pathname : imageLabels){
            log.info(pathname);
        }

        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int numSamples = 600;
        int batchSize = 750;
        int iterations = 200;
        int seed = 123;
        int listenerFreq = batchSize;

        RecordReader recordReader = new ImageRecordReader(numRows, numColumns, true, imageLabels);

        recordReader.initialize(new FileSplit(new File(dataPath)));

        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, -1, imageLabels.size());

        // check dataset
        /*
        int counter = 0;
        while(iter.hasNext()){
            DataSet tds = iter.next();
            INDArray showm = tds.getFeatureMatrix();
            System.out.println(showm);
            log.info("----------");
            counter ++;
        }
        log.info(Integer.toString(counter));
        */

        DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();

        INDArray showme;
        log.info("Split data...");
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(numSamples, new Random(seed));
        DataSet train = testAndTrain.getTrain();
        showme = train.getFeatureMatrix();
        System.out.println(showme);
        System.out.println("the matrix len is " + showme.rows() + " mpl " + showme.columns());
        log.info("----------------");
        DataSet test = testAndTrain.getTest();
        showme = test.getFeatureMatrix();
        System.out.println(showme);
        System.out.println("the matrix len is " + showme.rows() + " mpl " + showme.columns());
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;


        //DataSetIterator iter = new MnistDataSetIterator(batchSize, numSamples, true);



        log.info("Build model...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .weightInit(WeightInit.XAVIER)
                .iterations(iterations)
                .momentum(0.5)  // ???
                .momentumAfter(Collections.singletonMap(3, 0.9))  // ???
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list()
                .layer(0, new RBM.Builder().nIn(numRows * numColumns).nOut(500)
                                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                                .visibleUnit(RBM.VisibleUnit.BINARY)
                                .hiddenUnit(RBM.HiddenUnit.BINARY)
                                .build())
                .layer(1, new RBM.Builder().nIn(500).nOut(250)
                                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                                .visibleUnit(RBM.VisibleUnit.BINARY)
                                .hiddenUnit(RBM.HiddenUnit.BINARY)
                                .build())
                .layer(2, new RBM.Builder().nIn(250).nOut(200)
                                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                                .visibleUnit(RBM.VisibleUnit.BINARY)
                                .hiddenUnit(RBM.HiddenUnit.BINARY)
                                .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
                                .nIn(200).nOut(outputNum).build())
                .pretrain(true).backprop(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train modal...");
        model.fit(train);

        log.info("Evaluate model...");
        Evaluation eval = new Evaluation(outputNum);
        INDArray output = model.output(test.getFeatureMatrix());

        for(int i = 0; i < output.rows(); ++ i){
            String actual = test.getLabels().getRow(i).toString().trim();
            String predicted = output.getRow(i).toString().trim();
            log.info("actual" + actual + "vs predicted " + predicted);
        }

        eval.eval(test.getLabels(), output);
        log.info(eval.stats());
        log.info("****************Example finished********************");

    }
}
