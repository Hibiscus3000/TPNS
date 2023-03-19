package ru.nsu.fit.tpnn.lab1.panel.matrix;

import ru.nsu.fit.tpnn.lab1.GBC;
import ru.nsu.fit.tpnn.lab1.selection.Selection;

import javax.swing.*;
import java.awt.*;
import java.math.BigDecimal;
import java.util.List;

import static javax.swing.ScrollPaneConstants.*;

public class MatrixPanel extends JPanel {

    private static final Dimension preferredLabelDimension = new Dimension(200, 100);

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
            label.setPreferredSize(preferredLabelDimension);
            horizontalLabelPanel.add(label, new GBC(i, 0, GridBagConstraints.BOTH));
        }

        JPanel verticalLabelPanel = new JPanel();
        verticalLabelPanel.setLayout(new GridBagLayout());
        int numberOfRows = matrix.length;
        for (int i = 0; i < numberOfRows; ++i) {
            JLabel label = new JLabel(targets.get(i).getValueName());
            label.setPreferredSize(preferredLabelDimension);
            verticalLabelPanel.add(label, new GBC(0, i, GridBagConstraints.VERTICAL));
        }

        JPanel matrixPanel = new JPanel();
        matrixPanel.setLayout(new GridBagLayout());
        for (int i = 0; i < numberOfColumns; ++i) {
            for (int j = 0; j < numberOfRows; ++j) {
                JLabel label = new JLabel(String.valueOf(matrix[j][i]));
                label.setPreferredSize(preferredLabelDimension);
                matrixPanel.add(label, new GBC(j, i, GridBagConstraints.BOTH));
            }
        }

        JScrollPane horizontalLabelPane = new JScrollPane(horizontalLabelPanel, VERTICAL_SCROLLBAR_NEVER,
                HORIZONTAL_SCROLLBAR_NEVER);
        JScrollPane verticalLabelPane = new JScrollPane(verticalLabelPanel, VERTICAL_SCROLLBAR_NEVER,
                HORIZONTAL_SCROLLBAR_NEVER);
        JScrollPane matrixPane = new JScrollPane(matrixPanel, VERTICAL_SCROLLBAR_AS_NEEDED,
                HORIZONTAL_SCROLLBAR_AS_NEEDED);
        matrixPane.getHorizontalScrollBar().addAdjustmentListener(e ->
                horizontalLabelPane.getHorizontalScrollBar()
                        .setValue(matrixPane.getHorizontalScrollBar().getValue()));
        matrixPane.getVerticalScrollBar().addAdjustmentListener(e ->
                verticalLabelPane.getVerticalScrollBar()
                        .setValue(matrixPane.getVerticalScrollBar().getValue()));

        //verticalLabelPane.setMaximumSize(new Dimension(100, 600));
        //verticalLabelPanel.setMaximumSize(new Dimension(100, 600));
        //verticalLabelPane.setPreferredSize(new Dimension(100, 600));
        //verticalLabelPanel.setPreferredSize(new Dimension(100, 600));

        add(new JLabel(metricName));

        JPanel gridBagPanel = new JPanel();
        gridBagPanel.setLayout(new GridBagLayout());
        gridBagPanel.add(horizontalLabelPane, new GBC(1, 0, selections.size(), 1,
                GridBagConstraints.BOTH, 3));
        gridBagPanel.add(matrixPane, new GBC(1, 1, selections.size(), targets.size(),
                GridBagConstraints.BOTH, 3));
        gridBagPanel.add(verticalLabelPane, new GBC(0, 1, 1, targets.size(),
                GridBagConstraints.VERTICAL, 3));
        //verticalLabelPane.revalidate();
        //gridBagPanel.revalidate();

        add(gridBagPanel);
    }
}
