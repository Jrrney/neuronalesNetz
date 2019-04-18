package dietrich.jenny.informatik.neuronales.netz.internal;

import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * @author Jenny Dietrich
 *
 *         15.04.2019
 */
public class FileUtil {

	/**
	 * Liest alle Labels von einer IDX1-UBYTE Datei aus: <br>
	 * <ol>
	 * <li>2049. Steht für Labels</li>
	 * <li>Anzahl der Labels.</li>
	 * <li>Folgend alle Labels</li>
	 * </ol>
	 * 
	 * @param filePath
	 *            Gibt den Pfad an, wo die Datei mit den Labels hinterlegt ist.
	 * @return Ein Array von allen Labels.
	 * @throws IOException
	 *             falls die Datei nicht gefunden wurde oder sie keine Labels enthält.
	 */
	public static int[] readLabels(Path filePath) throws IOException {
		ByteBuffer bb = ByteBuffer.wrap(Files.readAllBytes(filePath));
		if (bb.getInt() != 2049) {
			throw new IOException("Keine Labels!");
		}

		int numberOfLabels = bb.getInt();
		int[] labels = new int[numberOfLabels];

		for (int i = 0; i < numberOfLabels; i++) {
			labels[i] = bb.get() & 0xFF;
		}
		return labels;
	}

	/**
	 * Liest alle Bild Daten von einer IDX3-UBYTE Datei aus: <br>
	 * <ol>
	 * <li>2051. Steht für Bild-Daten</li>
	 * <li>Anzahl der Bilder.</li>
	 * <li>Anzahl der Zeilen, die jedes Bild besitzt.</li>
	 * <li>Anzahl der Spalten, die jedes Bild besitzt.</li>
	 * <li>Folgend alle Daten für die Bilder</li>
	 * </ol>
	 * 
	 * @param filePath
	 *            Gibt den Pfad an, wo die Datei mit den Bilder-Daten hinterlegt ist.
	 * @return Eine Liste von Matrizen mit den jeweiligen Bild-Daten.
	 * @throws IOException
	 *             falls die Datei nicht gefunden wurde oder sie keine Bild-Daten enthält.
	 */
	public static List<int[][]> readImages(Path filePath) throws IOException {
		ByteBuffer bb = ByteBuffer.wrap(Files.readAllBytes(filePath));
		if (bb.getInt() != 2051) {
			throw new IOException("Keine Bilder!");
		}

		int anzahlImg = bb.getInt();
		int rows = bb.getInt();
		int columns = bb.getInt();
		List<int[][]> img = new ArrayList<>();

		for (int i = 0; i < anzahlImg; i++) {
			int[][] curImage = new int[rows][columns];
			for (int row = 0; row < rows; row++) {
				for (int column = 0; column < columns; ++column) {
					curImage[row][column] = bb.get() & 0xFF;
				}
			}
			img.add(curImage);
		}

		return img;
	}

	/**
	 * Liest alle Daten von einem Bild aus. <br>
	 * 
	 * @param filePath
	 *            Gibt den Pfad an, wo das Bild hinterlegt ist.
	 * @return Eine Matrize mit den jeweiligen Bild-Daten.
	 * @throws IOException
	 *             falls die Datei nicht gefunden wurde.
	 */
	public static int[][] readImage(Path filePath) throws IOException {
		BufferedImage bi = ImageIO.read(filePath.toFile());
		int rows = bi.getHeight();
		int columns = bi.getWidth();
		int[][] image = new int[rows][columns];

		for (int column = 0; column < columns; column++) {
			for (int row = 0; row < rows; row++) {
				image[row][column] = 255 - (bi.getRGB(column, row) & 0xFF);
			}
		}
		return image;
	}

	public static List<RealMatrix> readWeights(Path filePath, int numLayers) throws IOException, ClassNotFoundException {
		List<RealMatrix> weights = new ArrayList<>();
		try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath.toFile()))) {
			for (int i = 0; i < numLayers - 1; i++) {
				double[][] readedWeights = (double[][]) ois.readObject();
				weights.add(MatrixUtils.createRealMatrix(readedWeights));
			}
		}
		return weights;
	}

	public static void writeWeights(Path filePath, List<RealMatrix> weights) throws FileNotFoundException, IOException {
		try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath.toFile()))) {
			weights.forEach(matrix -> {
				try {
					oos.writeObject(matrix.getData());
				} catch (IOException e) {
					e.printStackTrace();
				}
			});
		}
	}
}
