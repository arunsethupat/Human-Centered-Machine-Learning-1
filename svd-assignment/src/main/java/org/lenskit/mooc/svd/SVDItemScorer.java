package org.lenskit.mooc.svd;

import com.google.common.collect.Ordering;
import it.unimi.dsi.fastutil.longs.LongSet;
import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.lenskit.api.Result;
import org.lenskit.api.ResultList;
import org.lenskit.api.ResultMap;
import org.lenskit.basic.AbstractItemScorer;
import org.lenskit.bias.BiasModel;
import org.lenskit.data.dao.DataAccessObject;
import org.lenskit.results.BasicResult;
import org.lenskit.results.Results;
import org.lenskit.util.collections.LongUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * SVD-based item scorer.
 */
public class SVDItemScorer extends AbstractItemScorer {
    private static final Logger logger = LoggerFactory.getLogger(SVDItemScorer.class);
    private final SVDModel model;
    private final BiasModel baseline;
    private final DataAccessObject dao;

    /**
     * Construct an SVD item scorer using a model.
     * @param m The model to use when generating scores.
     * @param dao The data access object.
     * @param bias The baseline bias model (providing means).
     */
    @Inject
    public SVDItemScorer(SVDModel m, DataAccessObject dao,
                         BiasModel bias) {
        model = m;
        baseline = bias;
        this.dao = dao;
    }

    public SVDModel getModel() {
        return model;
    }

    /**
     * Score items in a vector. The key domain of the provided vector is the
     * items to score, and the score method sets the values for each item to
     * its score (or unsets it, if no score can be provided). The previous
     * values are discarded.
     *
     * @param user   The user ID.
     * @param items The items to score
     */
    @Nonnull
    @Override
    public ResultMap scoreWithDetails(long user, @Nonnull Collection<Long> items) {
        RealVector userFeatures = model.getUserVector(user);
        if (userFeatures == null) {
            logger.debug("unknown user {}", user);
            return Results.newResultMap();
        }

        LongSet itemSet = LongUtils.asLongSet(items);
        List<Result> results = new ArrayList<>();
        // TODO Compute the predictions
        // TODO Add the predicted offsets to the baseline score
        // TODO Store the results in 'results'
        //DiagonalMatrix featureDiagMatrix = new DiagonalMatrix(model.getFeatureWeights().toArray());
        for(Long item : itemSet){
            RealVector itemFeatures = model.getItemVector(item);
            double predictedVal = userFeatures.ebeMultiply(model.getFeatureWeights()).dotProduct(itemFeatures);
            //adding baseline
            predictedVal+=(baseline.getIntercept() + baseline.getUserBias(user) + baseline.getItemBias(item));
            double popBlendedRating = (1 - model.getPopularityWeight())*predictedVal + model.getPopularityWeight()*model.getItemPopularity(item);
            results.add(new BasicResult(item, popBlendedRating));
        }
//        ResultList orderedList = getTopNResults(10, results);
//        for(Result res : orderedList) {
//            System.out.println("Item Id: " + res.getId() + " Rating: " + res.getScore() + " Popularity : " + model.getItemPopularity(res.getId()));
//        }
        return Results.newResultMap(results);
    }
    @Nonnull
    private ResultList getTopNResults(int n, Iterable<Result> scores) {
        Ordering<Result> ord = Results.scoreOrder();
        List<Result> topN;
        if (n < 0) {
            topN = ord.reverse().immutableSortedCopy(scores);
        } else {
            topN = ord.greatestOf(scores, n);
        }
        return Results.newResultList(topN);
    }
}
