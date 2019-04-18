package dietrich.jenny.informatik.neuronales.netz.internal;

import static dietrich.jenny.informatik.neuronales.netz.internal.Util.initializeRandomWeights;
import static dietrich.jenny.informatik.neuronales.netz.internal.Util.multiplyElements;
import static dietrich.jenny.informatik.neuronales.netz.internal.Util.scalar;
import static org.apache.commons.math3.linear.MatrixUtils.createRealMatrix;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * @author Jenny Dietrich
 *
 *         15.04.2019
 */
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
	private final static int epochs = 5;
	private final static double learning_rate = 0.1;

	private static RealMatrix WEIGHTS_INPUT_HIDDEN;
	private static RealMatrix WEIGHTS_HIDDEN_OUTPUT;

	private static int[] labels;
	private static double[][] scaledImages;
	private static double[][] roatedScaledImages_1;
	private static double[][] roatedScaledImages_2;

	public static void main(String[] args) throws ClassNotFoundException, IOException {
		initializeMaterial("./train-labels.idx1-ubyte", "./train-images.idx3-ubyte");
		initializeWeights("./weights");
		train();
	}

	public static void initializeMaterial(String labelFilePath, String imagesFilePath) throws IOException {
		labels = FileUtil.readLabels(Paths.get(labelFilePath));
		List<int[][]> trainingImages = FileUtil.readImages(Paths.get(imagesFilePath));
		scaledImages = new double[trainingImages.size()][];
		for (int i = 0; i < trainingImages.size(); i++)
			scaledImages[i] = scale(Util.flat(trainingImages.get(i)));

		roatedScaledImages_1 = new double[trainingImages.size()][];
		for (int i = 0; i < trainingImages.size(); i++)
			roatedScaledImages_1[i] = scale(Util.flat(Util.rotateImg(trainingImages.get(i), 10)));

		roatedScaledImages_2 = new double[trainingImages.size()][];
		for (int i = 0; i < trainingImages.size(); i++)
			roatedScaledImages_2[i] = scale(Util.flat(Util.rotateImg(trainingImages.get(i), -10)));
	}

	public static void initializeWeights(String weightsFilePath) throws ClassNotFoundException, IOException {
		Path weightsPath = Paths.get(weightsFilePath);
		if (Files.exists(weightsPath)) {
			List<RealMatrix> readWeights = FileUtil.readWeights(weightsPath, 3);
			WEIGHTS_INPUT_HIDDEN = readWeights.get(0);
			WEIGHTS_HIDDEN_OUTPUT = readWeights.get(1);
		} else {
			WEIGHTS_INPUT_HIDDEN = createRealMatrix(HIDDEN_NODES, INPUT_NODES);
			WEIGHTS_HIDDEN_OUTPUT = createRealMatrix(OUTPUT_NODES, HIDDEN_NODES);
			WEIGHTS_INPUT_HIDDEN = initializeRandomWeights(WEIGHTS_INPUT_HIDDEN, Math.pow(INPUT_NODES, -0.5));
			WEIGHTS_HIDDEN_OUTPUT = initializeRandomWeights(WEIGHTS_HIDDEN_OUTPUT, Math.pow(HIDDEN_NODES, -0.5));
		}
	}

	private static void train() {
		for (int e = 0; e < epochs; e++) {
			System.out.println("running epoch: " + (e + 1));
			for (int i = 0; i < labels.length; i++) {
				if (i % 100 == 0)
					System.out.println("running: " + (i));
				RealMatrix inputs = MatrixUtils.createColumnRealMatrix(scaledImages[i]);
				RealMatrix targets = createTarget(labels[i]);
				train(inputs, targets);

				inputs = MatrixUtils.createColumnRealMatrix(roatedScaledImages_1[i]);
				train(inputs, targets);

				inputs = MatrixUtils.createColumnRealMatrix(roatedScaledImages_2[i]);
				train(inputs, targets);
			}
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
}