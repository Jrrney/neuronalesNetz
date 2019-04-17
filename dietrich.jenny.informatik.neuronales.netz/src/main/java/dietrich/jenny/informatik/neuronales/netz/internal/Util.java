package dietrich.jenny.informatik.neuronales.netz.internal;

import static java.lang.System.arraycopy;
import static org.apache.commons.math3.linear.MatrixUtils.createRealMatrix;

import java.util.Random;
import java.util.function.Function;

import org.apache.commons.math3.linear.RealMatrix;

public abstract class Util {

	private final static Random RANDOM = new Random();

	/**
	 * Rechnet einen Wert mit der Sigmoid-Aktivierungsfunktion für einen Wert x aus.
	 */
	public static double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	/**
	 * Skaliert eine Matrix.
	 * 
	 * @param matrix
	 *            Die zu verändernde Matrix.
	 * @param function
	 *            Wie die Matrix verändert werden soll.
	 * @return Die aktualisierte Matrix.
	 */
	public static RealMatrix scalar(RealMatrix matrix, Function<Double, Double> function) {
		return foreach(matrix, function);
	}

	public static RealMatrix foreach(RealMatrix matrix, Function<Double, Double> function) {
		int rows = matrix.getRowDimension();
		int columns = matrix.getColumnDimension();
		RealMatrix result = createRealMatrix(rows, columns);
		for (int row = 0; row < rows; row++)
			for (int column = 0; column < columns; column++)
				result.setEntry(row, column, function.apply(matrix.getEntry(row, column)));
		return result;
	}

	public static RealMatrix multiplyElements(RealMatrix matrixA, RealMatrix matrixB) throws IllegalArgumentException {
		int rows = matrixA.getRowDimension();
		int columns = matrixA.getColumnDimension();
		RealMatrix multipliedMatrix = createRealMatrix(rows, columns);

		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				multipliedMatrix.setEntry(row, column, matrixA.getEntry(row, column) * matrixB.getEntry(row, column));
			}
		}
		return multipliedMatrix;
	}

	/**
	 * Wandelt eine Matrix in ein Array um, indem man jede Zeile nimmt und sie ans Ende packt.
	 * 
	 * @param i
	 *            Die umzuwandelnde Matrix.
	 * @return Ein Array, was jede Zeile von der Matrix enthält.
	 */
	public static int[] flat(int[][] i) {
		int[] result = new int[i.length * i[0].length];
		for (int row = 0; row < i.length; row++) {
			int[] curRow = i[row];
			arraycopy(curRow, 0, result, row * curRow.length, curRow.length);
		}
		return result;
	}

	public static RealMatrix initializeRandomWeights(RealMatrix matrix, double standardDeviation) {
		return foreach(matrix, in -> RANDOM.nextGaussian() * standardDeviation);
	}

}