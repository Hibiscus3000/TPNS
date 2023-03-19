package ru.nsu.fit.tpnn.lab1;

import java.awt.*;

public class GBC extends GridBagConstraints {

    public GBC(int gridX, int gridY) {
        this(gridX, gridY, 1, 1, BOTH, 5);
    }

    public GBC(int gridX, int gridY, int fill) {
        this(gridX, gridY, 1, 1, fill, 5);
    }

    public GBC(int gridX, int gridY, int gridWidth, int gridHeight, int fill, int inset) {
        gridx = gridX;
        gridy = gridY;
        gridwidth = gridWidth;
        gridheight = gridHeight;
        insets = new Insets(inset, inset, inset, inset);
        this.fill = fill;
        if (BOTH != fill) {
            anchor = WEST;
        }
        weightx = 100;
        weighty = 100;
    }
}
