import org.lenskit.api.ItemScorer
import org.lenskit.mooc.svd.PopularityWeight
import org.lenskit.mooc.svd.SVDItemScorer
import org.lenskit.bias.*

// Set up item scorer
bind ItemScorer to SVDItemScorer.class
set PopularityWeight to 25
bind BiasModel to GlobalBiasModel