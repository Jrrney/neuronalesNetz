package dietrich.jenny.informatik.neuronales.netz.internal;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;

import org.apache.commons.math3.linear.RealMatrix;

/**
 * @author Jenny Dietrich
 *
 *         15.04.2019
 */
public class UI extends JPanel {

	private static final long serialVersionUID = 1L;

	private int[][] currentImage;
	private int predictedLabel;
	private int expectedLabel;

	private static JFrame f;

	public UI() throws ClassNotFoundException, IOException {
		NeuralNetwork.initializeWeights("./weights");
		f = new JFrame();
		f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		f.setBounds(100, 100, 380, 380);
		JPanel contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		contentPane.setLayout(new BorderLayout(0, 0));

		contentPane.add(this, BorderLayout.CENTER);

		f.setContentPane(contentPane);
		f.setVisible(true);
	}

	@Override
	public void paintComponent(Graphics g) {
		if (currentImage != null) {

			Graphics2D g2d = (Graphics2D) g;
			setSize(280, 280);
			int w = this.getWidth();
			int h = this.getHeight();

			g2d.setColor(Color.WHITE);
			g2d.fillRect(0, 0, w, h);

			int[][] img = this.currentImage;

			for (int row = 0; row < img.length; row++) {
				int[] element = img[row];
				for (int col = 0; col < element.length; col++) {
					int x = col * w / 28;
					int y = row * h / 28;
					if (element[col] > 0) {
						int color = 255 - element[col];
						g2d.setColor(new Color(color, color, color));
						g2d.fillRect(x, y, w / 28 + 1, h / 28 + 1);
					}
				}
			}

			g2d.setFont(new Font("Arial", Font.BOLD, 80));
			if (expectedLabel == predictedLabel) {
				g2d.setPaint(Color.GREEN);
				g2d.drawString(String.valueOf(this.predictedLabel), 10, 70);
			} else {
				g2d.setPaint(Color.RED);
				g2d.drawString(this.predictedLabel + " (" + expectedLabel + ")", 10, 70);
			}

		}
	}

	public void test(int label, int[][] image) throws InterruptedException {
		RealMatrix result = NeuralNetwork.query(NeuralNetwork.scale(Util.flat(image)));
		this.predictedLabel = NeuralNetwork.getMatchingOutput(result);
		this.currentImage = image;
		this.expectedLabel = label;
		f.repaint();
		Thread.sleep(3000);
	}

	public void test(int[] labels, List<int[][]> images) throws InterruptedException {
		for (int i = 0; i < labels.length; i++) {
			int[][] img = images.get(i);
			RealMatrix result = NeuralNetwork.query(NeuralNetwork.scale(Util.flat(img)));
			this.predictedLabel = NeuralNetwork.getMatchingOutput(result);
			this.currentImage = img;
			this.expectedLabel = labels[i];
			f.repaint();
			Thread.sleep(3000);
		}
	}

	public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException, InterruptedException {

		int label = 5;
		int[][] image = FileUtil.readImage(Paths.get("./Unbenannt.png"));

		int[] labels = FileUtil.readLabels(Paths.get("./t10k-labels.idx1-ubyte"));
		List<int[][]> images = FileUtil.readImages(Paths.get("./t10k-images.idx3-ubyte"));

		UI p = new UI();
		p.test(label, image);
		p.test(labels, images);

	}

}