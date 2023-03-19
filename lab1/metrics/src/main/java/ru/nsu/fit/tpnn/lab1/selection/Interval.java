package ru.nsu.fit.tpnn.lab1.selection;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Interval {

    private final int id;
    private final BigDecimal leftBorder;
    private final BigDecimal rightBorder;

    private final boolean rightmostInterval;

    private final Map<Integer, BigDecimal> values = new HashMap<>();

    public Interval(int id, BigDecimal leftBorder, BigDecimal rightBorder, boolean rightmostInterval) {
        this.id = id;
        this.leftBorder = leftBorder;
        this.rightBorder = rightBorder;

        this.rightmostInterval = rightmostInterval;
    }

    public void addValue(int id, Value value) {
        values.put(id, value.getValue());
        value.setIntervalId(this.id);
    }

    public boolean isIn(BigDecimal value) {
        return !rightmostInterval && 0 >= leftBorder.compareTo(value) && 0 < rightBorder.compareTo(value)
                || rightmostInterval && 0 > leftBorder.compareTo(value) && 0 <= rightBorder.compareTo(value);
    }

    public int getNumberOfValues() {
        return values.size();
    }

    public List<Integer> getValuesIds() {
        return new ArrayList<>(values.keySet());
    }
}
