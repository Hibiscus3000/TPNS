package ru.nsu.fit.tpnn.lab1;

import java.awt.*;

public class GBC extends GridBagConstraints {

    public GBC(int gridX, int gridY) {
        this(gridX, gridY, 1, 1);
    }

    public GBC(int gridX, int gridY, int gridWidth, int gridHeight) {
        this(gridX, gridY, gridWidth, gridHeight, 100, 100);
    }

    public GBC(int gridX, int gridY, int gridWidth, int gridHeight, int weightX, int weightY) {
        gridx = gridX;
        gridy = gridY;
        gridwidth = gridWidth;
        gridheight = gridHeight;
        fill = BOTH;
        anchor = CENTER;
        weightx = weightX;
        weighty = weightY;
    }
}
