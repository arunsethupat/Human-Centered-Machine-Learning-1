/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lenskit.mooc.svd;

import it.unimi.dsi.fastutil.longs.LongSet;
import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Precision;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.*;
import org.codehaus.groovy.runtime.powerassert.SourceText;
import org.lenskit.data.dao.DataAccessObject;
import org.lenskit.data.entities.CommonAttributes;
import org.lenskit.data.ratings.Rating;
import org.lenskit.mooc.svd.SVDModelBuilder.MyRating;
import org.lenskit.util.keys.KeyIndex;

import java.util.HashMap;
import java.util.List;
import java.util.Random;

public class MySingularValueDecomposition {

    private final int m;

    private final int n;


    private HashMap<Long, Double> itemPopularity;
    private int featureCount ;
    RealMatrix userMatrix;
    RealMatrix itemMatrix;
    RealMatrix originalMatrix;
    double[][] ratingData ;
    List<MyRating> ratingList;
    KeyIndex itemIndex;
    double TARGET_POP = 2.5;
    public MySingularValueDecomposition(RealMatrix matrix, int featureCount, HashMap<Long, Double> itemPop, int noOfUsers, int noOfItems, KeyIndex itemIndex) {
        this.itemIndex = itemIndex;
        this.ratingList = ratingList;
        this.itemPopularity = itemPop;
        this.featureCount = featureCount;
        this.n = matrix.getColumnDimension();
        this.m = matrix.getRowDimension();
        userMatrix = MatrixUtils.createRealMatrix(noOfUsers, featureCount);
        Random rand = new Random();
        for(int i = 0; i< noOfUsers; i++) {
            for(int j = 0; j<featureCount; j++) {
                userMatrix.setEntry(i,j, rand.nextDouble()/10.0);
            }
        }
        itemMatrix = MatrixUtils.createRealMatrix(noOfItems, featureCount);
        for(int i = 0; i< noOfItems; i++) {
            for(int j = 0; j<featureCount; j++) {
                itemMatrix.setEntry(i,j, rand.nextDouble()/10.0);
            }
        }
        originalMatrix = matrix;
        ratingData = matrix.getData();
        computeSVD();
    }

    public void computeSVD(){

        int MAX_ITERATION =45;
        double alpha = 0.002;
        double beta = 0.02;
        double error = 0.0;
        double totalerror = 0.0;
        double weight = 1;
        double totalObjective = 0.0;
        int count = 0;
        for(int step = 1; step <= MAX_ITERATION; step++){
            totalerror = 0.0;
            count = 0;
            totalObjective = 0.0;
//            for(MyRating ratingObject: ratingList){
//                double rating = ratingObject.getRating();
//                long userIndex = ratingObject.getUserId();
//                long itemIndex = ratingObject.getItemId();
//
//                try{
//
//                    double[] u = userMatrix.getRow((int)userIndex);
//                    double[] v = itemMatrix.getRow((int)itemIndex);
//                    if(userIndex==704 && itemIndex==1227){
//                        System.out.print("\nU: ");
//                        for(int l=0; l<featureCount; l++){
//                            System.out.print(u[l]+" ");
//                        }
//                        System.out.print("\nV: ");
//                        for(int l=0; l<featureCount; l++){
//                            System.out.print(v[l]+" ");
//                        }
//                        System.out.println();
//                    }
//                    error = Math.pow((rating - dotProduct(u,v)),2); //
//                    totalerror += error;
//                    //Math.pow(dotProduct(u, v), 2);
//                    for(int k=0; k<featureCount; k++){
//                        double tempu = u[k];
//                        double tempv = v[k];
//                        u[k] = u[k] + alpha * (2 * error * v[k] - beta * u[k]);
//                        v[k] = v[k] + alpha * (2 * error * u[k] - beta * v[k]);
//                        if(u[k] >= Double.MAX_VALUE || v[k] >= Double.MAX_VALUE){
//                            System.out.println("Gotcha");
//                        }
//                        else if(u[k] <= Double.MIN_VALUE || v[k] <= Double.MIN_VALUE){
//                            System.out.println("Gotcha");
//                        }
//                    }
//                    //System.out.println("v[0]="+v[0]+"\tu[0]="+u[0]);
//                    userMatrix.setRow((int)userIndex, u);
//                    itemMatrix.setRow((int)itemIndex, v);
//                }catch (Exception ex){
//                    System.out.println(ex.getMessage());
//                }
////                System.out.println("Current since cell error: "+error);
//
//            }

            for(int i=0; i < ratingData.length; i++){
                for(int j=0; j<ratingData[0].length; j++){
                    double rating = ratingData[i][j];
                    double objective = 0.0;
                    if(rating != 0){
                        count++;
                        double[] u = userMatrix.getRow(i);
                        double[] v = itemMatrix.getRow(j);
//                        error = Math.pow((rating - dotProduct(u,v)),2); //
                        error = rating - dotProduct(u,v);
                        totalerror += Math.pow(error,2);
                        //Math.pow(dotProduct(u, v), 2);
                        long itemId = itemIndex.getKey(j);
                        //weight = weight + alpha * (Math.pow(itemPopularity.get(itemId)-TARGET_POP, 2) - Math.pow(error , 2));

                        for(int k=0; k<featureCount; k++){
                            u[k] = u[k] + alpha * (2 * weight * error * v[k] - beta * u[k]);
                            v[k] = v[k] + alpha * (2 * weight * error * u[k] - beta * v[k]);
                        }

                        userMatrix.setRow(i, u);
                        itemMatrix.setRow(j, v);

                        objective = weight * Math.pow(rating-dotProduct(u, v), 2) + (1 - weight) * Math.pow(itemPopularity.get(itemId) - TARGET_POP, 2);
                        totalObjective += objective;
                    }
//                    if(i==600)
//                        System.out.format("\n Jhakkasi:%d  j:%d error :%f",i, j, error);
//                    System.out.format("\n i:%d  j:%d error :%f",i, j, error);
//                    if(j%20 == 0){
//                        try {
//                            Thread.sleep(1000);
//                        } catch (InterruptedException e) {
//                            e.printStackTrace();
//                        }
//                    }
                }
                System.out.format("\n i:%d  Objective :%f  weight : %f",i, totalObjective, weight);


            }
//            RealMatrix predMatrix = userMatrix.multiply((itemMatrix).transpose());
//            computeError(predMatrix);
            System.out.format("\nIteration : %d over , Error : %f", step, Math.sqrt(totalerror / (count)));
            if(Math.abs(totalerror) <= 0.5)
                break;
        }

        System.out.format("\nTrained at Error : %f", totalerror);
    }

//    private double computeError(RealMatrix predMatrix){
//        predMatrix.
//    }
    private double dotProduct(double[] u, double[] v){
        double result = 0.0;
//        double[] tempU = new double[u.length];
//        for(int i=0; i< u.length; i++)
//            tempU[i] = u[i];
//        double[] tempV = new double[v.length];
//        for(int i=0; i< v.length; i++)
//            tempV[i] = v[i];
        for(int i=0; i < u.length; i++){
            result += (u[i] * v[i]);
        }
//        if(result >= Double.MAX_VALUE){
//            System.out.println(result);
//        }
//        else if(result <= Double.MIN_VALUE){
//            System.out.println(result);
//        }
        return result;
    }

    public RealMatrix getUserMatrix(){
        return userMatrix;
    }

    public RealMatrix getItemMatrix(){
        return itemMatrix;
    }

    /**
     * Returns the matrix U of the decomposition.
     * <p>U is an orthogonal matrix, i.e. its transpose is also its inverse.</p>
     * @return the U matrix
     * @see #getUT()
     */
//    public RealMatrix getU() {
//        // return the cached matrix
//        return cachedU;
//
//    }
//
//    /**
//     * Returns the transpose of the matrix U of the decomposition.
//     * <p>U is an orthogonal matrix, i.e. its transpose is also its inverse.</p>
//     * @return the U matrix (or null if decomposed matrix is singular)
//     * @see #getU()
//     */
//    public RealMatrix getUT() {
//        if (cachedUt == null) {
//            cachedUt = getU().transpose();
//        }
//        // return the cached matrix
//        return cachedUt;
//    }
//
//    /**
//     * Returns the diagonal matrix &Sigma; of the decomposition.
//     * <p>&Sigma; is a diagonal matrix. The singular values are provided in
//     * non-increasing order, for compatibility with Jama.</p>
//     * @return the &Sigma; matrix
//     */
//    public RealMatrix getS() {
//        if (cachedS == null) {
//            // cache the matrix for subsequent calls
//            cachedS = MatrixUtils.createRealDiagonalMatrix(singularValues);
//        }
//        return cachedS;
//    }
//
//    /**
//     * Returns the diagonal elements of the matrix &Sigma; of the decomposition.
//     * <p>The singular values are provided in non-increasing order, for
//     * compatibility with Jama.</p>
//     * @return the diagonal elements of the &Sigma; matrix
//     */
//    public double[] getSingularValues() {
//        return singularValues.clone();
//    }
//
//    /**
//     * Returns the matrix V of the decomposition.
//     * <p>V is an orthogonal matrix, i.e. its transpose is also its inverse.</p>
//     * @return the V matrix (or null if decomposed matrix is singular)
//     * @see #getVT()
//     */
//    public RealMatrix getV() {
//        // return the cached matrix
//        return cachedV;
//    }
//
//    /**
//     * Returns the transpose of the matrix V of the decomposition.
//     * <p>V is an orthogonal matrix, i.e. its transpose is also its inverse.</p>
//     * @return the V matrix (or null if decomposed matrix is singular)
//     * @see #getV()
//     */
//    public RealMatrix getVT() {
//        if (cachedVt == null) {
//            cachedVt = getV().transpose();
//        }
//        // return the cached matrix
//        return cachedVt;
//    }
//
//    /**
//     * Returns the n &times; n covariance matrix.
//     * <p>The covariance matrix is V &times; J &times; V<sup>T</sup>
//     * where J is the diagonal matrix of the inverse of the squares of
//     * the singular values.</p>
//     * @param minSingularValue value below which singular values are ignored
//     * (a 0 or negative value implies all singular value will be used)
//     * @return covariance matrix
//     * @exception IllegalArgumentException if minSingularValue is larger than
//     * the largest singular value, meaning all singular values are ignored
//     */
//    public RealMatrix getCovariance(final double minSingularValue) {
//        // get the number of singular values to consider
//        final int p = singularValues.length;
//        int dimension = 0;
//        while (dimension < p &&
//                singularValues[dimension] >= minSingularValue) {
//            ++dimension;
//        }
//
//        if (dimension == 0) {
//            throw new NumberIsTooLargeException(LocalizedFormats.TOO_LARGE_CUTOFF_SINGULAR_VALUE,
//                    minSingularValue, singularValues[0], true);
//        }
//
//        final double[][] data = new double[dimension][p];
//        getVT().walkInOptimizedOrder(new DefaultRealMatrixPreservingVisitor() {
//            /** {@inheritDoc} */
//            @Override
//            public void visit(final int row, final int column,
//                              final double value) {
//                data[row][column] = value / singularValues[row];
//            }
//        }, 0, dimension - 1, 0, p - 1);
//
//        RealMatrix jv = new Array2DRowRealMatrix(data, false);
//        return jv.transpose().multiply(jv);
//    }
//
//    /**
//     * Returns the L<sub>2</sub> norm of the matrix.
//     * <p>The L<sub>2</sub> norm is max(|A &times; u|<sub>2</sub> /
//     * |u|<sub>2</sub>), where |.|<sub>2</sub> denotes the vectorial 2-norm
//     * (i.e. the traditional euclidian norm).</p>
//     * @return norm
//     */
//    public double getNorm() {
//        return singularValues[0];
//    }
//
//    /**
//     * Return the condition number of the matrix.
//     * @return condition number of the matrix
//     */
//    public double getConditionNumber() {
//        return singularValues[0] / singularValues[n - 1];
//    }
//
//    /**
//     * Computes the inverse of the condition number.
//     * In cases of rank deficiency, the {@link #getConditionNumber() condition
//     * number} will become undefined.
//     *
//     * @return the inverse of the condition number.
//     */
//    public double getInverseConditionNumber() {
//        return singularValues[n - 1] / singularValues[0];
//    }
//
//    /**
//     * Return the effective numerical matrix rank.
//     * <p>The effective numerical rank is the number of non-negligible
//     * singular values. The threshold used to identify non-negligible
//     * terms is max(m,n) &times; ulp(s<sub>1</sub>) where ulp(s<sub>1</sub>)
//     * is the least significant bit of the largest singular value.</p>
//     * @return effective numerical matrix rank
//     */
//    public int getRank() {
//        int r = 0;
//        for (int i = 0; i < singularValues.length; i++) {
//            if (singularValues[i] > tol) {
//                r++;
//            }
//        }
//        return r;
//    }
//
//    /**
//     * Get a solver for finding the A &times; X = B solution in least square sense.
//     * @return a solver
//     */
//    public DecompositionSolver getSolver() {
//        return new Solver(singularValues, getUT(), getV(), getRank() == m, tol);
//    }
//
//    /** Specialized solver. */
//    private static class Solver implements DecompositionSolver {
//        /** Pseudo-inverse of the initial matrix. */
//        private final RealMatrix pseudoInverse;
//        /** Singularity indicator. */
//        private boolean nonSingular;
//
//        /**
//         * Build a solver from decomposed matrix.
//         *
//         * @param singularValues Singular values.
//         * @param uT U<sup>T</sup> matrix of the decomposition.
//         * @param v V matrix of the decomposition.
//         * @param nonSingular Singularity indicator.
//         * @param tol tolerance for singular values
//         */
//        private Solver(final double[] singularValues, final RealMatrix uT,
//                       final RealMatrix v, final boolean nonSingular, final double tol) {
//            final double[][] suT = uT.getData();
//            for (int i = 0; i < singularValues.length; ++i) {
//                final double a;
//                if (singularValues[i] > tol) {
//                    a = 1 / singularValues[i];
//                } else {
//                    a = 0;
//                }
//                final double[] suTi = suT[i];
//                for (int j = 0; j < suTi.length; ++j) {
//                    suTi[j] *= a;
//                }
//            }
//            pseudoInverse = v.multiply(new Array2DRowRealMatrix(suT, false));
//            this.nonSingular = nonSingular;
//        }
//
//        /**
//         * Solve the linear equation A &times; X = B in least square sense.
//         * <p>
//         * The m&times;n matrix A may not be square, the solution X is such that
//         * ||A &times; X - B|| is minimal.
//         * </p>
//         * @param b Right-hand side of the equation A &times; X = B
//         * @return a vector X that minimizes the two norm of A &times; X - B
//         * @throws org.apache.commons.math3.exception.DimensionMismatchException
//         * if the matrices dimensions do not match.
//         */
//        public RealVector solve(final RealVector b) {
//            return pseudoInverse.operate(b);
//        }
//
//        /**
//         * Solve the linear equation A &times; X = B in least square sense.
//         * <p>
//         * The m&times;n matrix A may not be square, the solution X is such that
//         * ||A &times; X - B|| is minimal.
//         * </p>
//         *
//         * @param b Right-hand side of the equation A &times; X = B
//         * @return a matrix X that minimizes the two norm of A &times; X - B
//         * @throws org.apache.commons.math3.exception.DimensionMismatchException
//         * if the matrices dimensions do not match.
//         */
//        public RealMatrix solve(final RealMatrix b) {
//            return pseudoInverse.multiply(b);
//        }
//
//        /**
//         * Check if the decomposed matrix is non-singular.
//         *
//         * @return {@code true} if the decomposed matrix is non-singular.
//         */
//        public boolean isNonSingular() {
//            return nonSingular;
//        }
//
//        /**
//         * Get the pseudo-inverse of the decomposed matrix.
//         *
//         * @return the inverse matrix.
//         */
//        public RealMatrix getInverse() {
//            return pseudoInverse;
//        }
//    }
}
