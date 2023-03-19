package ru.nsu.fit.tpnn.lab1.panel.matrix;

import ru.nsu.fit.tpnn.lab1.GBC;
import ru.nsu.fit.tpnn.lab1.selection.Selection;

import javax.swing.*;
import java.awt.*;
import java.util.List;

public class EntropyPanel extends JPanel {

    public EntropyPanel(List<Selection> selections) {
        super();
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

        JPanel matrixPanel = new JPanel();
        matrixPanel.setLayout(new GridBagLayout());

        int matrixSize = selections.size();
        for (int i = 0; i < matrixSize; ++i) {
            matrixPanel.add(new JLabel(selections.get(i).getValueName()), new GBC(i + 1, 0));
        }
        matrixPanel.add(new JLabel("name"), new GBC(0, 0));
        matrixPanel.add(new JLabel("entropy"), new GBC(0, 1));


        for (int i = 0; i < matrixSize; ++i) {
            matrixPanel.add(new JLabel(String.valueOf(selections.get(i).getEntropy())),
                    new GBC(i + 1, 1));
        }

        add(new JLabel("Entropy"));
        add(matrixPanel);
    }
}
