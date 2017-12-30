package org.lenskit.mooc.svd;


import org.apache.commons.math3.linear.*;
import org.lenskit.bias.BiasModel;
import org.lenskit.data.dao.DataAccessObject;
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
import java.util.HashMap;
import java.util.Map;

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
        featureCount = 25;
        popularityWeight = popWeight/100;
    }

    /**
     * Build the SVD model.
     *
     * @return A singular value decomposition recommender model.
     */
    @Override
    public SVDModel get() {
        // Create index mappings of user and item IDs.
        // You can use these to find row and columns in the matrix based on user/item IDs.
        KeyIndex userIndex = FrozenHashKeyIndex.create(dao.getEntityIds(CommonTypes.USER));
        KeyIndex itemIndex = FrozenHashKeyIndex.create(dao.getEntityIds(CommonTypes.ITEM));

        HashMap<Long, Double> itemPopularity =  calculateItemPopularity();
        RealMatrix matrix = createRatingMatrix(userIndex, itemIndex, itemPopularity);

        // Second, compute its factorization
        logger.info("factorizing matrix");
        SingularValueDecomposition svd = new SingularValueDecomposition(matrix);
        logger.info("decomposed matrix has rank {}", svd.getRank());

        // Third, truncate the decomposed matrix
        RealMatrix userMatrix = svd.getU();
        RealMatrix itemMatrix = svd.getV();
        RealVector weights = new ArrayRealVector(svd.getSingularValues());
        if (featureCount > 0) {
            logger.info("truncating matrix to {} features", featureCount);
            // TODO Use the getSubMatrix method to truncate the user and item matrices
            userMatrix = userMatrix.getSubMatrix(0, userMatrix.getRowDimension()-1, 0, featureCount-1);
            itemMatrix = itemMatrix.getSubMatrix(0, itemMatrix.getRowDimension()-1, 0, featureCount-1);
            weights = weights.getSubVector(0, featureCount);
        }

        return new SVDModel(userIndex, itemIndex,
                userMatrix, itemMatrix,
                weights, itemPopularity);
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
            int x=0;
            for(Map.Entry<Long, Double> itemPopPair : itemPop.entrySet()){
                if(itemPopPair.getKey() == 296)
                    x =0;
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
}
