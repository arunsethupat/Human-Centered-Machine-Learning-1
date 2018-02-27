package org.lenskit.mooc.svd;

import com.google.common.base.Throwables;
import it.unimi.dsi.fastutil.longs.LongSet;
import org.lenskit.LenskitConfiguration;
import org.lenskit.LenskitRecommender;
import org.lenskit.LenskitRecommenderEngine;
import org.lenskit.api.ItemRecommender;
import org.lenskit.api.Result;
import org.lenskit.api.ResultList;
import org.lenskit.config.ConfigHelpers;
import org.lenskit.data.dao.DataAccessObject;
import org.lenskit.data.dao.file.StaticDataSource;
import org.lenskit.data.entities.CommonAttributes;
import org.lenskit.data.entities.CommonTypes;
import org.lenskit.data.entities.Entity;
import org.lenskit.data.ratings.Rating;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import java.io.FileWriter;
/**
 * Demonstration app for LensKit. This application builds an item-item CF model
 * from a CSV file, then generates recommendations for a user.
 *
 * Usage: java org.grouplens.lenskit.hello.HelloLenskit ratings.csv user
 */
public class ConsoleRunner implements Runnable {
    private static final Logger logger = LoggerFactory.getLogger(ConsoleRunner.class);
    HashMap<Long, Integer> itemFreq = new HashMap<>();
    private static final String FILE_HEADER = "MovieId,Popularity,PopularityWeight,RecFrequency";
    private static final String NEW_LINE_SEPARATOR = "\n";
    private static final String COMMA_DELIMITER = ",";
    private final String freqFileName = "itemFreq_40.csv";

    public static void main(String[] args) {
        ConsoleRunner hello = new ConsoleRunner(args);
        try {
            hello.run();
        } catch (RuntimeException e) {
            System.err.println(e.toString());
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    private Path trainFile = Paths.get("data/movielens.yml");
    //private Path testFile = Paths.get("data/testFile.yml");
    private List<Long> users;

    public ConsoleRunner(String[] args) {
        users = new ArrayList<>(args.length);
        for (String arg: args) {
            users.add(Long.parseLong(arg));
        }
    }

    public void run() {
        // We first need to configure the data access.
        // We will load data from a static data source; you could implement your own DAO
        // on top of a database of some kind
        DataAccessObject trainDao;
        //DataAccessObject testDao;
        try {
            StaticDataSource trainData = StaticDataSource.load(trainFile);
            //StaticDataSource testData = StaticDataSource.load(testFile);
            // get the data from the DAO
            trainDao = trainData.get();
            //testDao = testData.get();
        } catch (IOException e) {
            logger.error("cannot load data", e);
            throw Throwables.propagate(e);
        }

        // Next: load the LensKit algorithm configuration
        LenskitConfiguration config = null;
        try {
            //config = ConfigHelpers.load(new File("etc/svd.groovy"));
//            config = ConfigHelpers.load(new File("etc/svd.groovy"));
            config = ConfigHelpers.load(new File("etc/svd.groovy"));
        } catch (IOException e) {
            throw new RuntimeException("could not load configuration", e);
        }


        // There are more parameters, roles, and components that can be set. See the
        // JavaDoc for each recommender algorithm for more information.

        // Now that we have a configuration, build a recommender engine from the configuration
        // and data source. This will compute the similarity matrix and return a recommender
        // engine that uses it.
        LenskitRecommenderEngine engine = LenskitRecommenderEngine.build(config, trainDao);

        logger.info("built recommender engine");

        // Finally, get the recommender and use it.
        try (LenskitRecommender rec = engine.createRecommender(trainDao)) {
            logger.info("obtained recommender from engine");
            // we want to recommend items
            ItemRecommender irec = rec.getItemRecommender();
            assert irec != null; // not null because we configured one
            // for users
            //List<Rating> testRating = testDao.query(Rating.class).get();
            List<Rating> ratings = trainDao.query(Rating.class).get();

            LongSet allUsers = trainDao.query(Rating.class).valueSet(CommonAttributes.USER_ID);

            for(Long testUser :allUsers){
                LongSet userRatedItems = trainDao.query(Rating.class).withAttribute(CommonAttributes.USER_ID, testUser).valueSet(CommonAttributes.ITEM_ID);
                System.out.format("Recommendations for user %d:\n", testUser);
                ResultList recs = irec.recommendWithDetails(testUser, 10, null, userRatedItems);
                displayRecommendations(recs);
                //updateFrequency(recs);
                System.out.println("-------------------------------------------");
                break;
            }
        }
        System.out.println("Total Items ever recommended : "+itemFreq.size());
//        writeHeader();
//        writeFrequency();
    }
    private void displayRecommendations (ResultList recs){
        for(Result res : recs){
            System.out.println("Movie Id : "+res.getId() + "\tScore : "+res.getScore());
        }
    }

    private void writeHeader(){
        FileWriter writer = null;
        try{
            writer = new FileWriter(freqFileName, true);
            writer.append(FILE_HEADER);
            writer.append(NEW_LINE_SEPARATOR);
        }catch (Exception ex){
            System.out.println("Error while writing header into csv file");
            ex.printStackTrace();
        }finally {
            try {
                writer.flush();
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void writeFrequency(){
        System.out.println("Dumping item frequencies into CSV ...");
        FileWriter writer = null;
        try {
            writer = new FileWriter(freqFileName, true);
            for(Map.Entry<Long, Integer> itemEntry : itemFreq.entrySet()){
                Long item = itemEntry.getKey();
                Integer count = itemEntry.getValue();

                writer.append(String.valueOf(item));
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
        System.out.println("Dumping done !");
    }
    private void updateFrequency(ResultList recs){
        for(Result res : recs){
            Integer count = itemFreq.get(res.getId());
            if(count == null)
                count = 0;
            itemFreq.put(res.getId(), count + 1);
        }
    }
}

