package ru.nsu.fit.tpnn.lab1.panel.matrix;

import ru.nsu.fit.tpnn.lab1.GBC;
import ru.nsu.fit.tpnn.lab1.selection.Selection;

import javax.swing.*;
import java.awt.*;
import java.util.List;

public class EntropyPanel extends JPanel {

    public EntropyPanel(List<Selection> selections, boolean withNoData) {
        super();
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

        JPanel matrixPanel = new JPanel();
        matrixPanel.setLayout(new GridBagLayout());

        int matrixSize = selections.size();
        for (int i = 0; i < matrixSize; ++i) {
            JLabel label = new JLabel(selections.get(i).getValueName());
            label.setPreferredSize(MatrixPanel.getPreferredLabelSize());
            matrixPanel.add(label, new GBC(i + 1, 0));
        }
        JLabel nameLabel = new JLabel("name");
        nameLabel.setPreferredSize(MatrixPanel.getPreferredLabelSize());
        matrixPanel.add(nameLabel, new GBC(0, 0));
        JLabel entropyLabel = new JLabel("entropy");
        entropyLabel.setPreferredSize(MatrixPanel.getPreferredLabelSize());
        matrixPanel.add(entropyLabel, new GBC(0, 1));


        for (int i = 0; i < matrixSize; ++i) {
            matrixPanel.add(new JLabel(String.valueOf(selections.get(i).getEntropy(withNoData))),
                    new GBC(i + 1, 1));
        }

        add(new JLabel("Entropy" + (withNoData ? " counting no data entries" : "")));
        add(new JScrollPane(matrixPanel));
        setPreferredSize(new Dimension(600, 200));
    }
}
