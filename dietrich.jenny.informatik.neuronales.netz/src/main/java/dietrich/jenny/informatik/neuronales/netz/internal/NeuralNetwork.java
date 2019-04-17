package dietrich.jenny.informatik.neuronales.netz.internal;

import static dietrich.jenny.informatik.neuronales.netz.internal.Util.initializeRandomWeights;
import static dietrich.jenny.informatik.neuronales.netz.internal.Util.multiplyElements;
import static dietrich.jenny.informatik.neuronales.netz.internal.Util.scalar;
import static org.apache.commons.math3.linear.MatrixUtils.createRealMatrix;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class NeuralNetwork {

	/**
	 * Anzahl der Input-Neuronen. Diese entsprechen der Anzahl der Pixel eines 28px * 28px Bildes.
	 */
	private final static int INPUT_NODES = 784;
	/**
	 * Anzahl der Neuronen eines Hidden-Layers. Kann beliebig gesetzt werden. (je nachdem, wie groﬂ
	 * das neuronale Netz werden soll)
	 */
	private final static int HIDDEN_NODES = 200;
	/**
	 * Anzahl der Output-Neuronen. Diese entsprechen jeweils den Zahlen 0-9.
	 */
	private final static int OUTPUT_NODES = 10;
	private final static int epochs = 2;
	private final static double learning_rate = 0.1;

	public static RealMatrix WEIGHTS_INPUT_HIDDEN;
	public static RealMatrix WEIGHTS_HIDDEN_OUTPUT;

	// Accuracy: 0.9766
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		Path filePath = Paths.get("./train-labels.idx1-ubyte");
		int[] labels = FileReader.readLabels(filePath);
		List<int[][]> images = FileReader.readImages(Paths.get("./train-images.idx3-ubyte"));

		double[][] scaledImages = new double[images.size()][];
		for (int i = 0; i < images.size(); i++)
			scaledImages[i] = scale(Util.flat(images.get(i)));

		// Alle Bilder um 10 Grad gedreht
		double[][] roatedScaledImages_1 = new double[images.size()][];
		for (int i = 0; i < images.size(); i++)
			roatedScaledImages_1[i] = scale(Util.flat(rotateImg(images.get(i), 10)));

		// Alle Bilder um -10 Grad gedreht
		double[][] roatedScaledImages_2 = new double[images.size()][];
		for (int i = 0; i < images.size(); i++)
			roatedScaledImages_2[i] = scale(Util.flat(rotateImg(images.get(i), -10)));

		Path weightsPath = Paths.get("./weights");
		if (Files.exists(weightsPath)) {
			try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream("./weights"))) {
				double[][] weightsInputHidden = (double[][]) ois.readObject();
				double[][] weightsHiddenOutput = (double[][]) ois.readObject();
				WEIGHTS_INPUT_HIDDEN = MatrixUtils.createRealMatrix(weightsInputHidden);
				WEIGHTS_HIDDEN_OUTPUT = MatrixUtils.createRealMatrix(weightsHiddenOutput);
			}
		} else {
			WEIGHTS_INPUT_HIDDEN = createRealMatrix(HIDDEN_NODES, INPUT_NODES);
			WEIGHTS_HIDDEN_OUTPUT = createRealMatrix(OUTPUT_NODES, HIDDEN_NODES);
			WEIGHTS_INPUT_HIDDEN = initializeRandomWeights(WEIGHTS_INPUT_HIDDEN, Math.pow(INPUT_NODES, -0.5));
			WEIGHTS_HIDDEN_OUTPUT = initializeRandomWeights(WEIGHTS_HIDDEN_OUTPUT, Math.pow(HIDDEN_NODES, -0.5));
		}

		for (int e = 0; e < epochs; e++) {
			System.out.println("running epoch: " + (e + 1));
			for (int i = 0; i < labels.length; i++) {
				if (i % 100 == 0)
					System.out.println("running: " + (i));
				RealMatrix inputs = MatrixUtils.createColumnRealMatrix(scaledImages[i]);
				RealMatrix targets = createTarget(labels[i]);
				train(inputs, targets);

				// inputs = MatrixUtils.createColumnRealMatrix(roatedScaledImages_1[i]);
				// train(inputs, targets);
				//
				// inputs = MatrixUtils.createColumnRealMatrix(roatedScaledImages_2[i]);
				// train(inputs, targets);
			}
		}

		int[] testLabels = FileReader.readLabels(Paths.get("./t10k-labels.idx1-ubyte"));
		List<int[][]> testImages = FileReader.readImages(Paths.get("./t10k-images.idx3-ubyte"));

		int correct = 0;
		for (int i = 0; i < testLabels.length; i++) {
			int correctLabel = testLabels[i];
			RealMatrix predict = query(scale(Util.flat(testImages.get(i))));
			int predictLabel = getMatchingOutput(predict);

			if (predictLabel == correctLabel) {
				correct++;
			}
		}

		System.out.println("Accuracy: " + correct / (double) testLabels.length);

		try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("./weights"))) {
			oos.writeObject(WEIGHTS_INPUT_HIDDEN.getData());
			oos.writeObject(WEIGHTS_HIDDEN_OUTPUT.getData());
		}

	}

	public static int getMatchingOutput(RealMatrix result) {
		double[][] data = result.getData();
		int indexMax = 0;
		for (int r = 0; r < data.length; r++)
			indexMax = data[r][0] > data[indexMax][0] ? r : indexMax;

		return indexMax;
	}

	public static RealMatrix query(double[] inputArray) {
		RealMatrix inputs = MatrixUtils.createColumnRealMatrix(inputArray);
		RealMatrix hiddenInputs = WEIGHTS_INPUT_HIDDEN.multiply(inputs);
		RealMatrix hiddenOutputs = scalar(hiddenInputs, Util::sigmoid);

		RealMatrix finalInputs = WEIGHTS_HIDDEN_OUTPUT.multiply(hiddenOutputs);
		RealMatrix finalOutputs = scalar(finalInputs, Util::sigmoid);
		return finalOutputs;
	}

	private static void train(RealMatrix inputs, RealMatrix targets) {
		RealMatrix hiddenInputs = WEIGHTS_INPUT_HIDDEN.multiply(inputs);
		RealMatrix hiddenOutputs = scalar(hiddenInputs, Util::sigmoid);

		RealMatrix finalInputs = WEIGHTS_HIDDEN_OUTPUT.multiply(hiddenOutputs);
		RealMatrix finalOutputs = scalar(finalInputs, Util::sigmoid);

		RealMatrix outputErrors = targets.subtract(finalOutputs);
		RealMatrix t1 = multiplyElements(outputErrors, finalOutputs);
		RealMatrix t2 = multiplyElements(t1, scalar(finalOutputs, in -> 1.0 - in));
		RealMatrix t3 = t2.multiply(hiddenOutputs.transpose());
		WEIGHTS_HIDDEN_OUTPUT = WEIGHTS_HIDDEN_OUTPUT.add(scalar(t3, in -> learning_rate * in));

		RealMatrix hiddenErrors = WEIGHTS_HIDDEN_OUTPUT.transpose().multiply(outputErrors);
		t1 = multiplyElements(hiddenErrors, hiddenOutputs);
		t2 = multiplyElements(t1, scalar(hiddenOutputs, in -> 1.0 - in));
		t3 = t2.multiply(inputs.transpose());
		WEIGHTS_INPUT_HIDDEN = WEIGHTS_INPUT_HIDDEN.add(scalar(t3, in -> learning_rate * in));
	}

	/**
	 * Skaliert den Input vom Input-Layer so, dass falls ein Wert 0 ist auf den Wert 0,001 gesetzt
	 * wird, damit die Ergebnisse nicht verf‰lscht werden.
	 * 
	 * @param img
	 *            Den zu skalierenden Input.
	 * @return Den skalierten Input.
	 */
	public static double[] scale(int[] img) {
		double[] result = new double[img.length];
		for (int i = 0; i < img.length; i++) {
			result[i] = img[i] / 255.0 * 0.999 + 0.001;
		}
		return result;
	}

	/**
	 * Erstellt den erwarteten Output.
	 * 
	 * @param label
	 * @return
	 */
	private static RealMatrix createTarget(int label) {
		RealMatrix target = MatrixUtils.createRealMatrix(10, 1);
		for (int i = 0; i < 10; i++) {
			target.setEntry(i, 0, i != label ? 0.001 : 0.999);
		}
		return target;
	}

	/**
	 * L‰sst die Inhalte einer Matrix um den Mittelpunkt drehen.
	 * 
	 * @param img
	 *            Die Matrix, die gedreht werden soll.
	 * @param grad
	 *            Wie viel Grad gedreht wird.
	 * @return Die gedrehte Matrix mit der originalen Grˆﬂe. Inhalte, die dar¸ber hinausreichen
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