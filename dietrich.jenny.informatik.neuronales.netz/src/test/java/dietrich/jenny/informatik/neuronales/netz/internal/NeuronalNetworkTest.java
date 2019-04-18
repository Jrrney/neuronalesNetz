/**
 * 
 */
package dietrich.jenny.informatik.neuronales.netz.internal;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

import org.apache.commons.math3.linear.RealMatrix;
import org.junit.Before;
import org.junit.Test;

/**
 * @author Jenny Dietrich
 *
 *         17.04.2019
 */
public class NeuronalNetworkTest {

	@Before
	public void setUp() {}

	@Test
	public void testName() throws IOException {
		int[] testLabels = FileUtil.readLabels(Paths.get("./t10k-labels.idx1-ubyte"));
		List<int[][]> testImages = FileUtil.readImages(Paths.get("./t10k-images.idx3-ubyte"));
		int correct = 0;

		for (int i = 0; i < testLabels.length; i++) {
			int correctLabel = testLabels[i];
			RealMatrix predict = NeuralNetwork.query(NeuralNetwork.scale(Util.flat(testImages.get(i))));
			int predictLabel = NeuralNetwork.getMatchingOutput(predict);

			if (predictLabel == correctLabel) {
				correct++;
			}
		}

		assertTrue((correct / (double) testLabels.length) > 0.95);
	}

}
