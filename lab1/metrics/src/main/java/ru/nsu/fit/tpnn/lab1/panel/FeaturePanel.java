package ru.nsu.fit.tpnn.lab1.panel;

import ru.nsu.fit.tpnn.lab1.GBC;
import ru.nsu.fit.tpnn.lab1.selection.Selection;

import javax.swing.*;
import java.awt.*;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public class FeaturePanel extends JPanel {

    private final List<Selection> selections;
    private final List<Selection> targets = new ArrayList<>();

    public FeaturePanel(List<Selection> selections) {
        super();
        this.selections = selections;
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        JPanel labelPanel = new JPanel();
        labelPanel.setLayout(new GridBagLayout());
        labelPanel.add(new JLabel("delete"), new GBC(0, 0));
        labelPanel.add(new JLabel("target"), new GBC(0, 1));
        labelPanel.add(new JLabel("name"), new GBC(0, 2));

        JPanel featuresSelectionPanel = new JPanel();
        featuresSelectionPanel.setLayout(new GridBagLayout());
        int x = 1;
        for (Selection selection : selections) {
            JCheckBox deleteCheckBox = new JCheckBox();
            JCheckBox targetCheckBox = new JCheckBox();
            targetCheckBox.addActionListener(e -> {
                if (targetCheckBox.isSelected()) {
                    targets.add(selection);
                } else {
                    targets.remove(selection);
                }
            });
            deleteCheckBox.addActionListener(e -> {
                if (deleteCheckBox.isSelected()) {
                    selections.remove(selection);
                    targets.remove(selection);
                } else {
                    selections.add(selection);
                    if (targetCheckBox.isSelected()) {
                        targets.add(selection);
                    }
                }
            });
            featuresSelectionPanel.add(deleteCheckBox, new GBC(x, 0));
            featuresSelectionPanel.add(targetCheckBox, new GBC(x, 1));
            featuresSelectionPanel.add(new JLabel(selection.getValueName()),
                    new GBC(x, 2));
            ++x;
        }

        JPanel boxPanel = new JPanel();
        boxPanel.setLayout(new BoxLayout(boxPanel, BoxLayout.X_AXIS));
        boxPanel.add(labelPanel);
        boxPanel.add(new JScrollPane(featuresSelectionPanel));

        add(boxPanel);
    }

    public BigDecimal[][] getGainRatios() {
        BigDecimal[][] gainRatios = new BigDecimal[targets.size()][selections.size()];
        for (int j = 0; j < targets.size(); ++j) {
            for (int i = 0; i < selections.size(); ++i) {
                gainRatios[j][i] = targets.get(j).getGainRatio(selections.get(i));
            }
        }
        return gainRatios;
    }

    public List<Selection> getTargets() {
        return targets;
    }
}
