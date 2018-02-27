package org.lenskit.mooc.svd;


import org.apache.commons.math3.linear.*;
import org.lenskit.bias.BiasModel;
import org.lenskit.data.dao.DataAccessObject;
import org.lenskit.data.entities.CommonAttributes;
import org.lenskit.data.entities.CommonTypes;
import org.lenskit.data.ratings.Rating;
import org.lenskit.inject.Transient;
import org.lenskit.util.io.ObjectStream;
import org.lenskit.util.keys.FrozenHashKeyIndex;
import org.lenskit.util.keys.KeyIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Provider;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Model builder that computes the SVD model.
 */
public class SVDModelBuilder implements Provider<SVDModel> {
    private static final Logger logger = LoggerFactory.getLogger(SVDModelBuilder.class);

    private final DataAccessObject dao;
    private final BiasModel baseline;
    private final int featureCount;
    private final double popularityWeight;

    /**
     * Construct the model builder.
     * @param dao The data access object.
     * @param bias The bias model to use as a baseline.
     * @param popWeight The weight given to popularity of Item.
     */
    @Inject
    public SVDModelBuilder(@Transient DataAccessObject dao,
                           @Transient BiasModel bias,
                           @PopularityWeight int popWeight) {
        this.dao = dao;
        baseline = bias;
        featureCount = 50;
        popularityWeight = popWeight/100.0;
        System.out.println("Popularity Weight Set to : "+popularityWeight);
    }

    /**
     * Build the SVD model.
     *
     * @return A singular value decomposition recommender model.
     */
    @Override
    public SVDModel get() {
        System.out.println("Building model for : "+popularityWeight);
        // Create index mappings of user and item IDs.
        // You can use these to find row and columns in the matrix based on user/item IDs.
        KeyIndex userIndex = FrozenHashKeyIndex.create(dao.getEntityIds(CommonTypes.USER));
        KeyIndex itemIndex = FrozenHashKeyIndex.create(dao.getEntityIds(CommonTypes.ITEM));

        HashMap<Long, Double> itemPopularity =  calculateItemPopularity();
//        writeItemPopularity(itemPopularity);
        RealMatrix matrix = createRatingMatrix(userIndex, itemIndex, itemPopularity);
        int noOfUsers = dao.query(Rating.class).valueSet(CommonAttributes.USER_ID).size();
        int noOfItems = dao.query(Rating.class).valueSet(CommonAttributes.ITEM_ID).size();
//        List<MyRating> ratingList = createRatingMatrix(userIndex, itemIndex, itemPopularity);
        // Second, compute its factorization
        logger.info("factorizing matrix at popularity weight : "+popularityWeight);
        MySingularValueDecomposition svd = new MySingularValueDecomposition(matrix, featureCount, itemPopularity, noOfUsers, noOfItems);
        RealMatrix userMatrix = svd.getUserMatrix();
        RealMatrix itemMatrix = svd.getItemMatrix();
        RealVector weights = MatrixUtils.createRealVector(new double[25]);


//        SingularValueDecomposition svd = new SingularValueDecomposition(matrix);
//        RealMatrix userMatrix = svd.getU();
//        RealMatrix itemMatrix = svd.getV();
//        RealVector weights = new ArrayRealVector(svd.getSingularValues());
//        if (featureCount > 0) {
//            logger.info("truncating matrix to {} features", featureCount);
//            // TODO Use the getSubMatrix method to truncate the user and item matrices
//            userMatrix = userMatrix.getSubMatrix(0, userMatrix.getRowDimension()-1, 0, featureCount-1);
//            itemMatrix = itemMatrix.getSubMatrix(0, itemMatrix.getRowDimension()-1, 0, featureCount-1);
//            weights = weights.getSubVector(0, featureCount);
//        }

//        System.out.format("User Matrix 10 x %d\n", featureCount);
//        for(int i=0; i<10; i++){
//            for(int j=0; j<featureCount; j++){
//                System.out.print(userMatrix.getEntry(i, j)+" ");
//            }
//            System.out.println();
//        }
//
//        System.out.format("Item Matrix 10 x %d \n", featureCount);
//        for(int i=0; i<10; i++){
//            for(int j=0; j<featureCount; j++){
//                System.out.print(itemMatrix.getEntry(i, j)+" ");
//            }
//            System.out.println();
//        }

        return new SVDModel(userIndex, itemIndex,
                userMatrix, itemMatrix,
                weights, itemPopularity, popularityWeight);
    }
    public HashMap<Long, Double> calculateItemPopularity(){
        HashMap<Long, Double> itemPop = new HashMap<>();
        try (ObjectStream<Rating> ratings = dao.query(Rating.class)
                .stream()) {
            // TODO Put this user's ratings into the matrix
            Double maxPop = -1.0;
            Double minPop = 99999999.0;
            for(Rating rating : ratings){
                Long itemId = rating.getItemId();
                Double count = itemPop.get(itemId);
                if(count == null)
                    count = 0.0;
                if(count+1 > maxPop)
                    maxPop = count+1;
                if(count+1 < minPop)
                    minPop = count+1;
                itemPop.put(itemId, count + 1);
            }
            for(Map.Entry<Long, Double> itemPopPair : itemPop.entrySet()){
                double normalizedPop = (itemPopPair.getValue() - minPop) / (maxPop - minPop) * 5;
                itemPopPair.setValue(normalizedPop);
            }

        }
        return itemPop;
    }
    /**
     * Build a rating residual matrix from the rating data.  Each user's ratings are
     * normalized by subtracting a baseline score (usually a mean).
     *
     * @param userIndex The index mapping of user IDs to row numbers.
     * @param itemIndex The index mapping of item IDs to column numbers.
     * @return A matrix storing the <i>normalized</i> user ratings.
     */
    private RealMatrix createRatingMatrix(KeyIndex userIndex, KeyIndex itemIndex, HashMap<Long, Double> itemPop) {
        final int nusers = userIndex.size();
        final int nitems = itemIndex.size();

        // Create a matrix with users on rows and items on columns
        logger.info("creating {} by {} rating matrix", nusers, nitems);
        RealMatrix matrix = MatrixUtils.createRealMatrix(nusers, nitems);

        try (ObjectStream<Rating> ratings = dao.query(Rating.class)
                .stream()) {
            // TODO Put this user's ratings into the matrix
            for(Rating rating : ratings){
                double normVal = rating.getValue();
                int uIndex = userIndex.getIndex(rating.getUserId());
                int iIndex = itemIndex.getIndex(rating.getItemId());
                if(normVal > 0){
                    double bias = baseline.getIntercept() + baseline.getUserBias(rating.getUserId()) + baseline.getItemBias(rating.getItemId());
                    normVal = normVal - bias;
                }
                Double pop = itemPop.get(rating.getItemId());
                normVal = (1-popularityWeight)*normVal + popularityWeight*pop;
                matrix.setEntry(uIndex, iIndex, normVal);
            }
        }
        return matrix;
    }

//    private List<MyRating> createRatingMatrix(KeyIndex userIndex, KeyIndex itemIndex, HashMap<Long, Double> itemPop) {
//        final int nusers = userIndex.size();
//        final int nitems = itemIndex.size();
//
//        logger.info("creating {} by {} rating matrix", nusers, nitems);
//        RealMatrix matrix = MatrixUtils.createRealMatrix(nusers, nitems);
//        List<MyRating> ratingList = new ArrayList<>();
//        try (ObjectStream<Rating> ratings = dao.query(Rating.class)
//                .stream()) {
//            // TODO Put this user's ratings into the matrix
//            for(Rating rating : ratings){
//                double normVal = rating.getValue();
//                int uIndex = userIndex.getIndex(rating.getUserId());
//                int iIndex = itemIndex.getIndex(rating.getItemId());
//                if(normVal > 0){
//                    double bias = baseline.getIntercept() + baseline.getUserBias(rating.getUserId()) + baseline.getItemBias(rating.getItemId());
//                    normVal = normVal - bias;
//                }
////                Double pop = itemPop.get(rating.getItemId());
////                normVal = (1-popularityWeight)*normVal + popularityWeight*pop;
//                MyRating rate = new MyRating(uIndex, iIndex, normVal);
//                ratingList.add(rate);
//                //matrix.setEntry(uIndex, iIndex, normVal);
//            }
//        }
//
//        Collections.shuffle(ratingList);
//        return ratingList;
//    }

    private void writeItemPopularity(HashMap<Long, Double> itemPopularity){
        System.out.println("Dumping item popularity into CSV ...");
        final String FILE_HEADER = "MovieId,Popularity";
        final String NEW_LINE_SEPARATOR = "\n";
        FileWriter writer = null;
        String freqFileName = "itemPopularity.csv";
        try {
            writer = new FileWriter(freqFileName, true);
            writer.append(FILE_HEADER);
            writer.append(NEW_LINE_SEPARATOR);
            for(Map.Entry<Long, Double> itemEntry : itemPopularity.entrySet()){
                Long item = itemEntry.getKey();
                Double pop = itemEntry.getValue();

                writer.append(String.valueOf(item));
                writer.append(",");
                writer.append(String.valueOf(pop));
                writer.append(",");
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
        System.out.println("Dumping done !");
    }

    class MyRating{
        public long getUserId() {
            return userId;
        }

        public long getItemId() {
            return itemId;
        }

        public double getRating() {
            return rating;
        }

        private long userId;
        private long itemId;
        private double rating;
        MyRating(long uid, long iid, double r){
            userId = uid;
            itemId = iid;
            rating = r;
        }
    }
}
