package dietrich.jenny.informatik.neuronales.netz.internal;

import static java.lang.System.arraycopy;
import static org.apache.commons.math3.linear.MatrixUtils.createRealMatrix;

import java.util.Random;
import java.util.function.Function;

import org.apache.commons.math3.linear.RealMatrix;

/**
 * @author Jenny Dietrich
 *
 *         15.04.2019
 */
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
		return scalar(matrix, in -> RANDOM.nextGaussian() * standardDeviation);
	}

	/**
	 * Lässt die Inhalte einer Matrix um den Mittelpunkt drehen.
	 * 
	 * @param img
	 *            Die Matrix, die gedreht werden soll.
	 * @param grad
	 *            Wie viel Grad gedreht wird.
	 * @return Die gedrehte Matrix mit der originalen Größe. Inhalte, die darüber hinausreichen
	 *         werden abgeschnitten.
	 */
	public static int[][] rotateImg(int[][] img, double grad) {
		double angle = Math.toRadians(grad);
		int height = img.length;
		int width = img[0].length;
		int middelH = height / 2;
		int middelW = width / 2;
		int[][] result = new int[img.length][img[0].length];

		int[] rotationMatrix2d = rotationMatrix2d(middelW, middelH, angle);
		double x0 = middelW - rotationMatrix2d[0];
		double y0 = middelH - rotationMatrix2d[1];

		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {

				int[] newKoordinate = rotationMatrix2d(x, y, angle);
				int xRot = (int) (newKoordinate[0] + x0);
				int yRot = (int) (newKoordinate[1] + y0);

				if (xRot >= 0 && yRot >= 0 && xRot <= width - 1 && yRot <= height - 1)
					result[y][x] = img[yRot][xRot];
			}
		}
		return result;
	}

	/**
	 * Berechnet einen Punkt, welcher um einen bestimmten Winkel vom Ausgangspunkt gedreht werden
	 * soll mit der zweidimensionalen DrehMatrix.
	 * 
	 * @param x
	 *            x-Koordinate
	 * @param y
	 *            y-Koordinate
	 * @param angle
	 *            Der Winkel, um den gedreht werden soll.
	 * @return
	 */
	private static int[] rotationMatrix2d(int x, int y, double angle) {
		int[] result = new int[2];
		result[0] = (int) (x * Math.cos(angle) + y * Math.sin(angle));
		result[1] = (int) (-x * Math.sin(angle) + y * Math.cos(angle));
		return result;
	}

}