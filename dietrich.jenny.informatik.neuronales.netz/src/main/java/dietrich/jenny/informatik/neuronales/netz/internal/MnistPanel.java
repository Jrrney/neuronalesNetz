package dietrich.jenny.informatik.neuronales.netz.internal;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;

import org.apache.commons.math3.linear.RealMatrix;

public class MnistPanel extends JPanel {

	private static final long serialVersionUID = 1L;

	private final int[] labels;
	private final List<int[][]> images;
	private int predictLabel;
	private int currentImage = -1;

	public MnistPanel(	int[] labels,
						List<int[][]> images) {
		this.labels = labels;
		this.images = images;
	}

	public boolean nextImage() {
		this.currentImage++;

		int[][] img = this.images.get(this.currentImage);
		RealMatrix result = NeuralNetwork.query(NeuralNetwork.scale(Util.flat(img)));
		this.predictLabel = NeuralNetwork.getMatchingOutput(result);

		return this.labels[this.currentImage] == this.predictLabel;
	}

	@Override
	public void paintComponent(Graphics g) {
		Graphics2D g2d = (Graphics2D) g;
		setSize(280, 280);
		int w = this.getWidth();
		int h = this.getHeight();

		g2d.setColor(Color.WHITE);
		g2d.fillRect(0, 0, w, h);

		int[][] img = this.images.get(this.currentImage);

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
		if (this.labels[this.currentImage] == this.predictLabel) {
			g2d.setPaint(Color.GREEN);
			g2d.drawString(String.valueOf(this.predictLabel), 10, 70);
		} else {
			g2d.setPaint(Color.RED);
			g2d.drawString(this.predictLabel + " (" + this.labels[this.currentImage] + ")", 10, 70);
		}

	}

	public static void main(String[] args) throws FileNotFoundException, IOException, InterruptedException, ClassNotFoundException {

		int[] labels = new int[3];
		labels[0] = 5;
		labels[1] = 5;
		labels[2] = 5;
		List<int[][]> images = new ArrayList<>();
		int[][] readImage = FileUtil.readImage(Paths.get("./Unbenannt.png"));
		images.add(readImage);
		images.add(Util.rotateImg(readImage, 10));
		images.add(Util.rotateImg(readImage, -10));
		// int[] labels = FileUtil.readLabels(Paths.get("./train-labels.idx1-ubyte"));
		// List<int[][]> images = FileUtil.readImages(Paths.get("./train-images.idx3-ubyte"));
		NeuralNetwork.initializeWeights("./weights");

		JFrame f = new JFrame();
		f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		f.setBounds(100, 100, 380, 380);
		JPanel contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		contentPane.setLayout(new BorderLayout(0, 0));

		MnistPanel p = new MnistPanel(labels, images);

		contentPane.add(p, BorderLayout.CENTER);

		f.setContentPane(contentPane);
		f.setVisible(true);

		for (int i = 0; i < 10000; i++) {
			boolean ok = p.nextImage();
			f.repaint();

			if (!ok) {
				Thread.sleep(5000);
			} else {
				Thread.sleep(1000);
			}
		}
	}

}