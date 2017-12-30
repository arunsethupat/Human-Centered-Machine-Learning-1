import org.lenskit.api.ItemScorer
import org.lenskit.bias.*
import org.lenskit.knn.NeighborhoodSize
import org.lenskit.knn.item.ItemItemScorer
import org.lenskit.mooc.svd.PopularityWeight
import org.lenskit.mooc.svd.SVDItemScorer
import org.lenskit.transform.normalize.MeanCenteringVectorNormalizer
import org.lenskit.transform.normalize.VectorNormalizer


// test different SVD sizes
for (popWeight in [0, 20, 30, 40, 50]) {
    algorithm("SVD") {
        attributes["PopularityWeight"] = popWeight
        attributes["Bias"] = "global"
        bind ItemScorer to SVDItemScorer
        set PopularityWeight to popWeight
        // compute SVD of offsets from global mean
        bind BiasModel to GlobalBiasModel
    }
}

