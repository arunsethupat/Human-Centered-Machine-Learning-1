package org.lenskit.mooc.svd;

import it.unimi.dsi.fastutil.longs.LongList;
import it.unimi.dsi.fastutil.longs.LongSet;
import org.apache.commons.math3.linear.RealVector;
import org.codehaus.groovy.runtime.powerassert.SourceText;
import org.grouplens.lenskit.util.statistics.MeanAccumulator;
import org.lenskit.LenskitRecommender;
import org.lenskit.api.Recommender;
import org.lenskit.eval.traintest.AlgorithmInstance;
import org.lenskit.eval.traintest.DataSet;
import org.lenskit.eval.traintest.TestUser;
import org.lenskit.eval.traintest.metrics.MetricColumn;
import org.lenskit.eval.traintest.metrics.MetricResult;
import org.lenskit.eval.traintest.metrics.TypedMetricResult;
import org.lenskit.eval.traintest.recommend.ListOnlyTopNMetric;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.inject.Inject;
import java.io.IOException;
import java.util.HashMap;
import java.io.FileWriter;
import java.util.Map;
import java.util.Scanner;

public class ILSMetric extends ListOnlyTopNMetric<ILSMetric.Context> {

//    private HashMap<Double, SVDModel> modelMap;
    private HashMap<Double, HashMap<Long, Integer>> popWeightToItemRecFreq;
    private final String freqFileName = "itemRecFrequency.csv";
    private static final String FILE_HEADER = "MovieId,Popularity,PopularityWeight,RecFrequency";
    private static final String NEW_LINE_SEPARATOR = "\n";
    private static final String COMMA_DELIMITER = ",";

//    private SVDModel model = null;
    @Inject
    public ILSMetric() {
        super(ILSMetric.UserResult.class, ILSMetric.AggregateResult.class);
//        itemRecFrequency = new HashMap<>();
        popWeightToItemRecFreq = new HashMap<>();
//        FileWriter writer = null;
//        try{
//            writer = new FileWriter(freqFileName, true);
//            writer.append(FILE_HEADER);
//            writer.append(NEW_LINE_SEPARATOR);
//        }catch (Exception ex){
//            System.out.println("Error while writing header into csv file");
//            ex.printStackTrace();
//        }finally {
//            try {
//                writer.flush();
//                writer.close();
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//        }
    }

    @Nullable
    @Override
    public Context createContext(AlgorithmInstance algorithm, DataSet dataSet, Recommender recommender) {
        return new Context(dataSet.getAllItems(), (LenskitRecommender) recommender);
    }

    @Nonnull
    @Override
    synchronized public MetricResult getAggregateMeasurements(ILSMetric.Context context) {
//        SVDModel model = ((SVDItemScorer)context.recommender.getItemScorer()).getModel();
//        SVDModel model = context.recommender.get(SVDModel.class);
//        writeItemRecFrequency(model);
//        Double keyPopWeight = model.getPopularityWeight();
//        System.out.println("Total Items : "+popWeightToItemRecFreq.get(keyPopWeight).size() + "for pop weight : "+keyPopWeight);
//        System.out.println("Removing item recommendation hashmap for pop weight : "+keyPopWeight);
//        popWeightToItemRecFreq.remove(keyPopWeight);
        return new ILSMetric.AggregateResult(context);
    }

    @Nonnull
    @Override
    public MetricResult measureUser(TestUser user, int targetLength, LongList recommendations, Context context) {
        SVDModel model = context.recommender.get(SVDModel.class);
//        System.out.println("measureUser :: pop weight : "+model.getPopularityWeight());
        Double cosine = 0.0;
        for(Long item1 : recommendations){
            for(Long item2 : recommendations){
                RealVector item1Vec = model.getItemVector(item1);
                RealVector item2Vec = model.getItemVector(item2);
                cosine += calculateCosineSimilarity(item1Vec, item2Vec);
            }

//            updateRecFrequency(item1, model);
        }
        cosine/=2;
        ILSMetric.UserResult result = new ILSMetric.UserResult(cosine);
        context.addUser(result);
        return result;
    }

    synchronized private void updateRecFrequency(Long item, SVDModel model){
        Double keyPopWeight = model.getPopularityWeight();
        HashMap<Long, Integer> itemRecFreq = popWeightToItemRecFreq.get(keyPopWeight);
        if(itemRecFreq == null)
            itemRecFreq = new HashMap<>();

        Integer count = itemRecFreq.get(item);
        if(count == null)
            count = 0;

        itemRecFreq.put(item, count + 1);
        popWeightToItemRecFreq.put(keyPopWeight, itemRecFreq);
    }

    synchronized private void writeItemRecFrequency(SVDModel model){
        Double keyPopWeight = model.getPopularityWeight();
        if(popWeightToItemRecFreq == null || popWeightToItemRecFreq.get(keyPopWeight) == null)
            return;
        System.out.println("Dumping into CSV file for pop weight : "+model.getPopularityWeight() + " ...");
        FileWriter writer = null;
        try {
            writer = new FileWriter(freqFileName, true);
            for(Map.Entry<Long, Integer> itemFreq : popWeightToItemRecFreq.get(keyPopWeight).entrySet()){
                Long item = itemFreq.getKey();
                Integer count = itemFreq.getValue();
                Double pop = model.getItemPopularity(item);

                writer.append(String.valueOf(item));
                writer.append(COMMA_DELIMITER);
                writer.append(String.valueOf(pop));
                writer.append(COMMA_DELIMITER);
                writer.append(String.valueOf(keyPopWeight));
                writer.append(COMMA_DELIMITER);
                writer.append(String.valueOf(count));
                writer.append(NEW_LINE_SEPARATOR);
            }
        } catch (Exception e) {
            System.out.println("Error in CsvFileWriter !!!");
            e.printStackTrace();
        } finally {
            try {
                writer.flush();
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        System.out.println("Dumped into CSV file for pop weight : "+model.getPopularityWeight());
    }

    public static class UserResult extends TypedMetricResult {
        @MetricColumn("ILS")
        public final Double ilsValue;

        public UserResult(Double val) {
            ilsValue = val;
        }

        public Double getIlsValue() { return ilsValue; }

    }

    public static class AggregateResult extends TypedMetricResult {

        @MetricColumn("ILS")
        public final Double ils;

        public AggregateResult(ILSMetric.Context accum) {
            this.ils = accum.allMean.getMean();
        }
    }

    public static class Context {
        private final LongSet universe;
        private final MeanAccumulator allMean = new MeanAccumulator();
        private final LenskitRecommender recommender;

        Context(LongSet universe, LenskitRecommender recommender) {
            this.universe = universe;
            this.recommender = recommender;
        }

        void addUser(ILSMetric.UserResult ur) {
            allMean.add(ur.getIlsValue());
        }
    }

    private Double calculateCosineSimilarity(RealVector targetVector, RealVector candidateVector){
        Double cosineValue = 0D;
        cosineValue = targetVector.cosine(candidateVector);
        return cosineValue;
    }
}
