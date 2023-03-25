package ru.nsu.fit.tpnn.lab1;

import ru.nsu.fit.tpnn.lab1.selection.Selection;

import javax.swing.*;
import java.awt.*;
import java.awt.font.FontRenderContext;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.awt.geom.Rectangle2D;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static ru.nsu.fit.tpnn.lab1.selection.Value.roundingMode;
import static ru.nsu.fit.tpnn.lab1.selection.Value.scale;

public class GraphicDialog extends JDialog {

    private static final MathContext mathContext = new MathContext(10, RoundingMode.HALF_EVEN);

    private final List<Selection> selections;
    private final List<Selection> targets;
    private final BigDecimal[][] matrix;

    private static final Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE,
            Color.MAGENTA};

    private BigDecimal max;

    public GraphicDialog(String title, List<Selection> selections, List<Selection> targets,
                         BigDecimal[][] matrix) {
        setTitle(title);
        this.selections = selections;
        this.targets = targets;
        this.matrix = matrix;
        spotMinMax();
        setLayout(new FlowLayout());
        JPanel graphicAndTargetsPanel = new JPanel();
        graphicAndTargetsPanel.setLayout(new BoxLayout(graphicAndTargetsPanel, BoxLayout.Y_AXIS));
        graphicAndTargetsPanel.add(new GraphicPanel());
        graphicAndTargetsPanel.add(getTargetSignaturesPanel(targets));
        add(graphicAndTargetsPanel);
        JScrollPane scrollPane = new JScrollPane(getValueNamesPanel(selections));
        scrollPane.setPreferredSize(new Dimension(300, 550));
        add(scrollPane);
        printSorted();
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
    }

    private JPanel getTargetSignaturesPanel(List<Selection> targets) {
        JPanel targetSignaturesPanel = new JPanel();
        for (int i = 0; i < targets.size(); ++i) {
            JLabel targetNameLabel = new JLabel(targets.get(i).getValueName());
            targetNameLabel.setPreferredSize(new Dimension(100, 30));
            targetNameLabel.setHorizontalAlignment(SwingConstants.CENTER);
            targetNameLabel.setForeground(colors[i % colors.length]);
            targetSignaturesPanel.add(targetNameLabel);
        }
        return targetSignaturesPanel;
    }

    private void spotMinMax() {
        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[0].length; ++j) {
                if (null == max || 0 < matrix[i][j].compareTo(max)) {
                    max = matrix[i][j].setScale(scale, roundingMode);
                }
            }
        }
    }

    private JPanel getValueNamesPanel(List<Selection> selections) {
        JPanel valueNamesPanel = new JPanel();
        valueNamesPanel.setLayout(new BoxLayout(valueNamesPanel, BoxLayout.Y_AXIS));
        for (int i = 0; i < selections.size(); ++i) {
            valueNamesPanel.add(new JLabel(String.format("%d - %s", i, selections.get(i).getValueName())));
        }
        return valueNamesPanel;
    }

    private class GraphicPanel extends JPanel {

        private static final Dimension preferredSize = new Dimension(700, 500);

        private static final int pointDiameter = 5;

        private static final double startXPositionScale = 0.1;
        private static final double startYPositionScale = 0.9;
        private static final double endXPositionScale = 0.9;
        private static final double endYPositionScale = 0.05;

        private final int numberOfLevelsX;
        private float twoLevelsDistanceX;

        private int startXPosition;
        private int startYPosition;
        private int endXPosition;
        private int endYPosition;

        private Dimension size;

        public GraphicPanel() {
            numberOfLevelsX = matrix[0].length;
            setPreferredSize(preferredSize);
        }

        @Override
        public void paintComponent(Graphics g) {
            super.paintComponents(g);
            Graphics2D g2d = (Graphics2D) g;

            size = getSize();

            startXPosition = (int) (startXPositionScale * size.getWidth());
            startYPosition = (int) (startYPositionScale * size.getHeight());
            endXPosition = (int) (endXPositionScale * size.getWidth());
            endYPosition = (int) (endYPositionScale * size.getHeight());

            twoLevelsDistanceX = ((float) (endXPosition - startXPosition)) / numberOfLevelsX;

            drawAxes(g2d);

            for (int i = 0; i < matrix.length; ++i) {
                g2d.setStroke(new BasicStroke(2));
                g2d.setColor(colors[i % colors.length]);
                int j;
                double lastGraphicX = 0, lastGraphicY = 0;
                for (j = 0; j < matrix[0].length - 1; ++j) {
                    double graphicX = getGraphicX(j);
                    double graphicY = getGraphicY(matrix[i][j]);
                    Line2D line = new Line2D.Double(graphicX, graphicY,
                            lastGraphicX = getGraphicX(j + 1),
                            lastGraphicY = getGraphicY(matrix[i][j + 1]));
                    Ellipse2D.Double circle = new Ellipse2D.Double(graphicX - (double) pointDiameter / 2,
                            graphicY - (double) pointDiameter / 2, pointDiameter, pointDiameter);
                    g2d.draw(line);
                    g2d.fill(circle);
                }
                Ellipse2D.Double circle = new Ellipse2D.Double(lastGraphicX - (double) pointDiameter / 2,
                        lastGraphicY - (double) pointDiameter / 2, pointDiameter, pointDiameter);
                g2d.fill(circle);
            }
        }

        private void drawAxes(Graphics2D g2d) {
            g2d.setStroke(new BasicStroke(4));
            g2d.drawLine(startXPosition, startYPosition, startXPosition, endYPosition);
            g2d.drawLine(startXPosition, startYPosition, endXPosition, startYPosition);

            g2d.setStroke(new BasicStroke(1));

            FontRenderContext renderContext = g2d.getFontRenderContext();

            final int numberOfYSubAxes = 20;
            for (int i = 1; i <= numberOfYSubAxes; ++i) {
                double graphicY = startYPosition - (double) i * (startYPosition - endYPosition) / numberOfYSubAxes;
                Line2D.Double line = new Line2D.Double(startXPosition, graphicY, endXPosition, graphicY);
                String signature = String.valueOf(BigDecimal.valueOf(getRealY(graphicY)).setScale(2, RoundingMode.HALF_EVEN));
                Rectangle2D bounds = g2d.getFont().getStringBounds(signature, renderContext);
                g2d.setColor(Color.BLACK);
                g2d.drawString(signature, (int) (startXPosition - bounds.getWidth()) - 5,
                        (int) (graphicY + bounds.getHeight() / 2));
                g2d.setColor(new Color(200, 200, 200));
                g2d.draw(line);
            }

            for (int i = 0; i < numberOfLevelsX; ++i) {
                double graphicX = getGraphicX(i);
                Line2D.Double line = new Line2D.Double(graphicX, startYPosition, graphicX, endYPosition);
                String signature = String.valueOf(i);
                Rectangle2D bounds = g2d.getFont().getStringBounds(signature, renderContext);
                g2d.setColor(Color.BLACK);
                g2d.drawString(signature, (int) (graphicX - bounds.getWidth() / 2),
                        (int) (startYPosition + bounds.getHeight()));
                g2d.setColor(new Color(200, 200, 200));
                g2d.draw(line);
            }
        }

        private double getRealY(double graphicY) {
            return (startYPosition - graphicY) * max.doubleValue() / (startYPosition - endYPosition);
        }

        private double getGraphicY(BigDecimal realY) {
            try {
                return startYPosition
                        - realY.divide(max, mathContext).multiply(BigDecimal.valueOf(startYPosition - endYPosition)).doubleValue();
            } catch (ArithmeticException arithmeticException) {
                return startYPosition;
            }
        }

        private double getGraphicX(int elementIndex) {
            return startXPosition + (elementIndex + 1) * twoLevelsDistanceX;
        }
    }

    private void printSorted() {
        for (int i = 0; i < targets.size(); ++i) {
            System.out.println("Gain ratio for " + targets.get(i).getValueName());
            ArrayIndexComparator gainRatioComparator = new ArrayIndexComparator(matrix[i]);
            Integer[] indices = gainRatioComparator.createIndexArray();
            Arrays.sort(indices, gainRatioComparator);
            for (int j = 0;j < indices.length; ++j) {
                Integer index = indices[j];
                System.out.printf("\t%d) %s - %f%n",j,selections.get(index).getValueName(),matrix[i][index]);
            }
        }
    }

    public class ArrayIndexComparator implements Comparator<Integer>
    {
        private final BigDecimal[] array;

        public ArrayIndexComparator(BigDecimal[] array)
        {
            this.array = array;
        }

        public Integer[] createIndexArray()
        {
            Integer[] indexes = new Integer[array.length];
            for (int i = 0; i < array.length; i++)
            {
                indexes[i] = i;
            }
            return indexes;
        }

        @Override
        public int compare(Integer index1, Integer index2)
        {
            return array[index1].compareTo(array[index2]);
        }
    }
}
