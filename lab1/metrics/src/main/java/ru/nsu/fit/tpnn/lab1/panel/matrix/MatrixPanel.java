package ru.nsu.fit.tpnn.lab1.panel.matrix;

import ru.nsu.fit.tpnn.lab1.GBC;
import ru.nsu.fit.tpnn.lab1.GraphicDialog;
import ru.nsu.fit.tpnn.lab1.selection.Selection;

import javax.swing.*;
import java.awt.*;
import java.math.BigDecimal;
import java.util.List;

import static javax.swing.ScrollPaneConstants.*;

public class MatrixPanel extends JPanel {

    private static final Dimension preferredLabelSize = new Dimension(250, 30);

    public MatrixPanel(List<Selection> selections, List<Selection> targets,
                       BigDecimal[][] matrix,
                       String metricName) {
        super();
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

        if (0 == matrix[0].length) {
            return;
        }

        JPanel horizontalLabelPanel = new JPanel();
        horizontalLabelPanel.setLayout(new GridBagLayout());
        int numberOfColumns = matrix[0].length;
        for (int i = 0; i < numberOfColumns; ++i) {
            JLabel label = new JLabel(selections.get(i).getValueName());
            label.setPreferredSize(preferredLabelSize);
            horizontalLabelPanel.add(label, new GBC(i, 0));
        }

        JPanel verticalLabelPanel = new JPanel();
        verticalLabelPanel.setLayout(new GridBagLayout());
        int numberOfRows = matrix.length;
        for (int i = 0; i < numberOfRows; ++i) {
            JLabel label = new JLabel(targets.get(i).getValueName());
            label.setPreferredSize(preferredLabelSize);
            verticalLabelPanel.add(label, new GBC(0, i));
        }

        JPanel matrixPanel = new JPanel();
        matrixPanel.setLayout(new GridBagLayout());
        for (int i = 0; i < numberOfColumns; ++i) {
            for (int j = 0; j < numberOfRows; ++j) {
                JLabel label = new JLabel(String.valueOf(matrix[j][i]));
                label.setPreferredSize(preferredLabelSize);
                matrixPanel.add(label, new GBC(i, j));
            }
        }

        JScrollPane horizontalLabelPane = new JScrollPane(horizontalLabelPanel, VERTICAL_SCROLLBAR_NEVER,
                HORIZONTAL_SCROLLBAR_NEVER);
        JScrollPane verticalLabelPane = new JScrollPane(verticalLabelPanel, VERTICAL_SCROLLBAR_NEVER,
                HORIZONTAL_SCROLLBAR_ALWAYS);
        JScrollPane matrixPane = new JScrollPane(matrixPanel, VERTICAL_SCROLLBAR_ALWAYS,
                HORIZONTAL_SCROLLBAR_ALWAYS);
        matrixPane.getHorizontalScrollBar().addAdjustmentListener(e ->
                horizontalLabelPane.getHorizontalScrollBar()
                        .setValue(matrixPane.getHorizontalScrollBar().getValue()));
        matrixPane.getVerticalScrollBar().addAdjustmentListener(e ->
                verticalLabelPane.getVerticalScrollBar()
                        .setValue(matrixPane.getVerticalScrollBar().getValue()));

        add(new JLabel(metricName));

        JButton graphicButton = new JButton("graphic");
        graphicButton.addActionListener(e -> new GraphicDialog(metricName, selections, targets, matrix).setVisible(true));

        JPanel gridBagPanel = new JPanel();
        gridBagPanel.setLayout(new GridBagLayout());
        gridBagPanel.add(graphicButton, new GBC(0, 0, 20, 20, GridBagConstraints.NONE));
        gridBagPanel.add(horizontalLabelPane, new GBC(1, 0, selections.size(), 1,
                100, 20));
        gridBagPanel.add(matrixPane, new GBC(1, 1, selections.size(), targets.size(),
                100, 100));
        gridBagPanel.add(verticalLabelPane, new GBC(0, 1, 1, targets.size(),
                20, 100));

        add(gridBagPanel);

        setPreferredSize(new Dimension(600, Math.min(100 + 30 * numberOfRows, 400)));
    }

    public static Dimension getPreferredLabelSize() {
        return preferredLabelSize;
    }
}
